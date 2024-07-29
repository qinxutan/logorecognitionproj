import base64
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import io
from params.config import Config

class ImageConverter:
    def __init__(self, hashed_images):
        self.hashed_images = hashed_images

    def decode_and_save_images(self, url, directory):
        if "encoding_logo" in url:
            logo_data = base64.b64decode(url["encoding_logo"])
            logo_image = Image.open(io.BytesIO(logo_data))
            logo_image.save(os.path.join(directory, f"logo_{url['hash_logo']}.png"))

        if "encoding_favicon" in url:
            favicon_data = base64.b64decode(url["encoding_favicon"])
            favicon_image = Image.open(io.BytesIO(favicon_data))
            favicon_image.save(os.path.join(directory, f"favicon_{url['hash_favicon']}.png"))

        if "encoding_screenshot" in url:
            screenshot_data = base64.b64decode(url["encoding_screenshot"])
            screenshot_image = Image.open(io.BytesIO(screenshot_data))
            screenshot_image.save(os.path.join(directory, f"screenshot_{url['hash_screenshot']}.png"))

    def split_train_test(self):
        train_images, test_images = train_test_split(self.hashed_images, test_size=0.2, random_state=42)

        for url in train_images:
            self.decode_and_save_images(url, Config.training_dir)

        for url in test_images:
            self.decode_and_save_images(url, Config.testing_dir)

