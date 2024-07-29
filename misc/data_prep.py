import numpy as np
import random
import torch
import PIL.ImageOps
from torch.utils.data import Dataset
from params.config import Config
from PIL import Image
import os
import onnxruntime
import cv2
from paddleocr import PaddleOCR
import textdistance
import re


ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def do_ocr(image):
    try:
        result = ocr.ocr(image)
        if result is None or not isinstance(result, list) or len(result) == 0:
            return ''
        result = result[0]
        if result is None or not isinstance(result, list):
            return ''
        result = [line[1][0] for line in result if line and len(line) > 1]
        result = ' '.join(result)
    except Exception as e:
        return ''
    return result.lower()
    
def orb(img1, img2):
        try:
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            similar_regions = [i for i in matches if i.distance < 50]
            if len(matches) == 0:
                return 0, 0
        except:
            return 0, 0
        else:
            return len(similar_regions), len(matches)
        
def ocr_check(ocr1, ocr2):
    check_ocr = True
    text_sim = 0
    if len(ocr1) > 5 and len(ocr2) > 5:
        text_sim = textdistance.levenshtein.normalized_similarity(re.sub(r'\W+', '', ocr1), re.sub(r'\W+', '', ocr2))
        min_ocr_length = min(len(ocr1), len(ocr2))
        min_match_length = len(textdistance.lcsseq(re.sub(r'\W+', '', ocr1), re.sub(r'\W+', '', ocr2)))
        match_str_ratio = min_match_length / min_ocr_length
        if text_sim < 0.2 or match_str_ratio < 0.5 or (text_sim < 0.4 and len(ocr1) < 10 and len(ocr2) < 10):
            check_ocr = False
    else:
        check_ocr = False
    return text_sim if check_ocr else 0

def _load_image(path):
    try:
        if path.lower().endswith(".png"):
            src = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            trans_mask = src[:, :, 3] == 0
            src[trans_mask] = [255, 255, 255, 255]
            new_img = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
            new_img = cv2.resize(new_img, (200, 200))
        elif path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
            new_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            new_img = cv2.resize(new_img, (200, 200))
        else:
            return None
    except Exception as e:
        return None

    if new_img is None:
        return None
    if len(new_img.shape) == 2:
        new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
    return new_img

class SiameseNetworkDataset(Dataset):
    def load_neural_hash_model():
        try:
            nh_session = onnxruntime.InferenceSession('/home/q1-qinxu-int/htxinternship/logorecognition/neuralhash_model.onnx')
            nh_seed = open('/home/q1-qinxu-int/htxinternship/logorecognition/neuralhash_128x96_seed1.dat', 'rb').read()[128:]
            nh_seed = np.frombuffer(nh_seed, dtype=np.float32)
            nh_seed = nh_seed.reshape([96, 128])
            return nh_session, nh_seed
        except Exception as e:
            print(f"Error loading neural hash model: {e}")
            return None, None 

    def get_neural_hash(nh_session, nh_seed, im):
        if im is None:
            print("Image is None. Cannot proceed.")
            return "NULL"
        try:
            im = Image.fromarray(im)
            img = im.convert('RGB')
            image = img.resize([360, 360])

            arr = np.array(image).astype(np.float32) / 255.0
            arr = arr * 2.0 - 1.0
            arr = arr.transpose(2, 0, 1)
            arr = arr.reshape([1, 3, 360, 360])

            inputs = {nh_session.get_inputs()[0].name: arr}
            outs = nh_session.run(None, inputs)
            print("outs[0] shape:", outs[0].shape)

            hash_output = nh_seed.dot(outs[0].flatten())
            hash_bits = ''.join('1' if it >= 0 else '0' for it in hash_output)
            hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)

            return hash_hex
        except Exception as exc:
            print(f"Error in get_neural_hash: {exc}")
            return None

    def calculate_hash_similarity(hash1, hash2):
        if hash1 == None or hash2 == None:
            return 0.0

        distance = sum(1 for x, y in zip(hash1, hash2) if x != y)
        similarity = 1.0 - (distance / len(hash1))

        return similarity

    def __init__(self, root_dir, transform=None, should_invert=True):
        self.root_dir = root_dir
        self.transform = transform
        self.should_invert = should_invert
        self.image_paths = []

        image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Create full paths for each image file
        self.image_paths = [os.path.join(root_dir, img_name) for img_name in image_files]

        self.image_cache = {}

        # Seed random for reproducibility
        np.random.seed(123)
        random.seed(123)
        self.random_indexes = np.random.randint(len(self.image_paths), size=int(len(self.image_paths) / Config.train_batch_size) + 1)

    def __getitem__(self, index):

        self.nh_session, self.nh_seed = SiameseNetworkDataset.load_neural_hash_model()
        
        if self.nh_session is None or self.nh_seed is None:
            print("Failed to load neural hash model and seed. Exiting.")
            return
        
        img0_path = None
        should_get_same_class = random.randint(0, 1)

        img1_path = os.path.join(self.root_dir, random.choice(os.listdir(self.root_dir)))


        if should_get_same_class:
            while True:
                img0_path = os.path.join(self.root_dir, random.choice(os.listdir(self.root_dir)))
                if self._get_class_name(img0_path) == self._get_class_name(img1_path):
                    break
        else:
            while True:
                img0_path = os.path.join(self.root_dir, random.choice(os.listdir(self.root_dir)))
                if self._get_class_name(img0_path) != self._get_class_name(img1_path):
                    break

        img0_cv2 = _load_image(img0_path)
        img1_cv2 = _load_image(img1_path)

        ocr1 = do_ocr(img0_cv2)
        ocr2 = do_ocr(img1_cv2)
        ocr_similarity = ocr_check(ocr1, ocr2)

        orb_similarity, _ = orb(img0_cv2, img1_cv2)

        nh1 = SiameseNetworkDataset.get_neural_hash(self.nh_session, self.nh_seed, img0_cv2)
        nh2 = SiameseNetworkDataset.get_neural_hash(self.nh_session, self.nh_seed, img1_cv2)
        nh_similarity = SiameseNetworkDataset.calculate_hash_similarity(nh1, nh2)

        feature_vector = np.array([ocr_similarity, orb_similarity, nh_similarity])
        feature_vector = self.normalize_features(feature_vector)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        img0_pil = Image.open(img0_path).convert('RGB') 
        img1_pil = Image.open(img1_path).convert('RGB') 
        
        if self.transform is not None:
            img0 = self.transform(img0_pil).unsqueeze(0).to(device)
            img1 = self.transform(img1_pil).unsqueeze(0).to(device)

        class_name_1 = self._get_class_name(img0_path)
        class_name_2 = self._get_class_name(img1_path)
        label = torch.tensor(int(class_name_1 != class_name_2), dtype=torch.float32)

        return img0, img1, feature_vector, label
    
    def _get_class_name(self, path):
        # Extract the filename without extension
        filename = os.path.splitext(os.path.basename(path))[0]
        # Split the filename by hyphen and return the first part as class name
        return filename.split('-')[0]
    
    def __len__(self):  
        return len(self.image_paths)
    
    @staticmethod
    def normalize_features(features):
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return (features - mean) / std