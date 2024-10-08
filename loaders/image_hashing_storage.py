import base64
import boto3
import cairosvg
import imagehash
import io
import numpy as np
import onnxruntime
import os
import requests
import tempfile
from dotenv import load_dotenv
from loaders.extractor import Extractor
from PhishIntention.phishintention import PhishIntentionWrapper

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class ImageHashingStorage:
  def __init__(self, urls):
    self.main_urls = urls
    self.all_urls = []
  
  def extract_data(self):
    for url in self.main_urls:
      extractor = Extractor(url)
      result = extractor.run()
      self.all_urls.append(result["Main page"])
      for login_page in result["Login pages"]:
        self.all_urls.append(login_page)
  
  def download_logo(self, url, screenshot_path):
    try:
      if url["logo"]:
        logo_url = url["logo"]
        response = requests.get(logo_url)
        response.raise_for_status()
        data = response.content
        if logo_url.endswith(".svg"):
          image = cairosvg.svg2png(bytestring=data)
          with open("logo.png", "wb") as file:
            file.write(image)
          encoded_data = base64.b64encode(image).decode('utf-8')
        else:
          with open("logo.png", "wb") as file:
            file.write(data)
          encoded_data = base64.b64encode(data).decode('utf-8')
        url["encoding_logo"] = encoded_data
      else:
        self.get_logo_from_screenshot(url, screenshot_path)
    except:
      self.get_logo_from_screenshot(url, screenshot_path)
  
  def get_logo_from_screenshot(self, url, screenshot_path):
    phishintention_cls = PhishIntentionWrapper()
    phishintention_cls.test_orig_phishintention(screenshot_path)
    with open("logo.png", "rb") as file:
      data = file.read()
      encoded_data = base64.b64encode(data).decode('utf-8')
    url["encoding_logo"] = encoded_data
  
  def download_favicon(self, url):
    try:
      if url["favicon"]:
        favicon_url = url["favicon"]
        response = requests.get(favicon_url)
        response.raise_for_status()
        data = response.content
        if favicon_url.endswith(".svg"):
          image = cairosvg.svg2png(bytestring=data)
          encoded_data = base64.b64encode(image).decode('utf-8')
        else:
          encoded_data = base64.b64encode(data).decode('utf-8')
        url["encoding_favicon"] = encoded_data
    except:
      print("Error downloading favicon")
    
  def download_logo_favicon_screenshots(self):
    
    LOCAL_DL_PATH = ""
    def mkdtemp():
      # create temp directory
      tempfile.mkdtemp()
    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_experimental_option("prefs", {
      "profile.default_content_setting_values.automatic_downloads": 1,
      "download.default_directory": LOCAL_DL_PATH
    })
    options.add_argument('--no-sandbox')
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280x1696")
    options.add_argument("--single-process")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-dev-tools")
    options.add_argument("--no-zygote")
    options.add_argument(f"--user-data-dir={mkdtemp()}")
    options.add_argument(f"--data-path={mkdtemp()}")
    options.add_argument(f"--disk-cache-dir={mkdtemp()}")
    options.add_argument("--remote-debugging-port=9222")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    for url in self.all_urls:
      driver.get(url["url"])
      screenshot = driver.get_screenshot_as_png()
      encoded_screenshot = base64.b64encode(screenshot).decode('utf-8')
      url["encoding_screenshot"] = encoded_screenshot
      with open("screenshot.png", "wb") as file:
        file.write(screenshot)
      
      self.download_logo(url, "screenshot.png")
      self.download_favicon(url)
  
  def load_neural_hash_model(self):
    self.nh_session = onnxruntime.InferenceSession("neuralhash_model.onnx")
    self.nh_seed = open("neuralhash_128x96_seed1.dat", "rb").read()[128:]
    self.nh_seed = np.frombuffer(self.nh_seed, dtype=np.float32).reshape([96,128])
  
  def neural_hash_image(self, type, url):
    try:
      if url[f"encoding_{type}"]:
        encoded_data = url[f"encoding_{type}"]
        decoded_data = base64.b64decode(encoded_data)
        image = Image.open(io.BytesIO(decoded_data))
        image = image.convert("RGB")
        image = image.resize([360,360])
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = arr.transpose(2,0,1).reshape([1,3,360,360])
        inputs = {self.nh_session.get_inputs()[0].name: arr}
        outputs = self.nh_session.run(None, inputs)
        hash_output = self.nh_seed.dot(outputs[0].flatten()) 
        hash_bits = ''.join(['1' if x >= 0 else '0' for x in hash_output])
        hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)
        url[f"hash_{type}"] = hash_hex
    except:
      return None
  
  def hash_logo_favicon_screenshots(self):
    for url in self.all_urls:
      self.load_neural_hash_model()
      self.neural_hash_image("logo", url)
      self.neural_hash_image("favicon", url)
      
      decoded_screenshot = base64.b64decode(url["encoding_screenshot"])
      screenshot = Image.open(io.BytesIO(decoded_screenshot))
      screenshot_hash = imagehash.dhash(screenshot)
      url["hash_screenshot"] = screenshot_hash
    
  def store_logo_images_favicon_screenshots(self):
    load_dotenv()
    access_key = os.getenv("access_key")
    secret_access_key = os.getenv("secret_access_key")
    dyanmo = boto3.resource(service_name='dynamodb', region_name='ap-southeast-1', aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)
    url_table = dyanmo.Table('ddb-htx-le-devizapp-imagehashes')

    for url in self.all_urls:
      if url["logo"] and url["favicon"]:
        url_table.put_item(Item={'url': url["url"], 'encoding_logo': url["encoding_logo"], 'encoding_favicon': url["encoding_favicon"], 'encoding_screenshot': url["encoding_screenshot"], 'hash_logo': str(url["hash_logo"]), 'hash_favicon': str(url["hash_favicon"]), 'hash_screenshot': str(url["hash_screenshot"])})
      elif url["logo"]:
        url_table.put_item(Item={'url': url["url"], 'encoding_logo': url["encoding_logo"], 'encoding_screenshot': url["encoding_screenshot"], 'hash_logo': str(url["hash_logo"]), 'hash_screenshot': str(url["hash_screenshot"])})
      elif url["favicon"]:
        url_table.put_item(Item={'url': url["url"], 'encoding_favicon': url["encoding_favicon"], 'encoding_screenshot': url["encoding_screenshot"], 'hash_favicon': str(url["hash_favicon"]), 'hash_screenshot': str(url["hash_screenshot"])})
      else:
        url_table.put_item(Item={'url': url["url"], 'encoding_screenshot': url["encoding_screenshot"], 'hash_screenshot': str(url["hash_screenshot"])})
      
  def run(self):
    self.extract_data()
    self.download_logo_favicon_screenshots()
    self.hash_logo_favicon_screenshots()
    self.store_logo_images_favicon_screenshots()
    print(self.all_urls)
    return self.all_urls
    
if __name__ == "__main__":
  urls = input("Enter the urls you want to hash: ").split()
  hashStorage = ImageHashingStorage(urls)
  hashStorage.run()

  