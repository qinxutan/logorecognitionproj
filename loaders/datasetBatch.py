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
    
# Function to compute ORB feature matching
def orb(img1, img2):
    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        similar_regions = [i for i in matches if i.distance < 30]
        if len(matches) == 0:
            return 0, 0
    except:
        return 0, 0
    else:
        return len(similar_regions), len(matches)

# Function to check OCR similarity
def ocr_check(ocr1, ocr2):
    check_ocr = True
    text_sim = 0
    if len(ocr1) > 1 and len(ocr2) > 1:
        ocr1_clean = re.sub(r'\W+', '', ocr1)
        ocr2_clean = re.sub(r'\W+', '', ocr2)
        text_sim = textdistance.levenshtein.normalized_similarity(ocr1_clean, ocr2_clean)

        if len(ocr1_clean) > 2 and len(ocr2_clean) > 2:
            if ocr1_clean in ocr2_clean or ocr2_clean in ocr1_clean:
                text_sim = 1.0

        min_ocr_length = min(len(ocr1_clean), len(ocr2_clean))

        if min_ocr_length == 0:
            return 0

        min_match_length = len(textdistance.lcsseq(ocr1_clean, ocr2_clean))
        match_str_ratio = min_match_length / min_ocr_length
        
        if text_sim < 0.2 or match_str_ratio < 0.6 or (text_sim < 0.3 and len(ocr1_clean) < 10 and len(ocr2_clean) < 10):
            check_ocr = False
    else:
        check_ocr = False
    return text_sim if check_ocr else 0

def detect_roi(image):
    """Detect and return the region of interest (ROI) in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image  # Return the original image if no contour is detected

    # Assume the largest contour is the ROI
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    roi = image[y:y+h, x:x+w]

    return roi

# Function to load and preprocess an image
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

def calculate_similarity_scores(ocr_sim, orb_sim, nh_sim):
        # Define weight for each feature
        weights = {'ocr': 0.4, 'orb': 0.3, 'nh': 0.3}

        # Adjusting threshold values
        ocr_threshold = 0.1
        orb_threshold = 0.1
        nh_threshold = 0.1
        
        # Weighted score calculation
        score = (weights['ocr'] * (ocr_sim if ocr_sim > ocr_threshold else 0) +
                weights['orb'] * (orb_sim if orb_sim > orb_threshold else 0) +
                weights['nh'] * (nh_sim if nh_sim > nh_threshold else 0))
        
        return score

class SiameseNetworkDataset(Dataset):
    def load_neural_hash_model():
        try:
            nh_session = onnxruntime.InferenceSession('/Users/qinxutan/Documents/htxinternship/logorecognition/neuralhash_model.onnx')
            nh_seed = open('/Users/qinxutan/Documents/htxinternship/logorecognition/neuralhash_128x96_seed1.dat', 'rb').read()[128:]
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

        # List all image files with .jpg, .jpeg, or .png extensions recursively in root_dir
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)
                    try:
                        # Attempt to open the image to verify it's a valid image file
                        Image.open(file_path).verify()
                        self.image_paths.append(file_path)
                    except (PIL.UnidentifiedImageError, IOError, SyntaxError) as e:
                        print(f"Skipping {file_path} due to error: {e}")

        self.image_cache = {}

        # Seed random for reproducibility
        np.random.seed(123)
        random.seed(123)
        self.random_indexes = np.random.randint(len(self.image_paths), size=int(len(self.image_paths) / Config.train_batch_size) + 1)

    def __getitem__(self, index):
        self.nh_session, self.nh_seed = SiameseNetworkDataset.load_neural_hash_model()

        if self.nh_session is None or self.nh_seed is None:
            print("Failed to load neural hash model and seed. Exiting.")
            return None

        while True:
            try:
                img0_path = None
                should_get_same_class = random.randint(0, 1)

                img1_class = random.choice(os.listdir(self.root_dir))
                img1_class_path = os.path.join(self.root_dir, img1_class)

                img1_path = os.path.join(img1_class_path, random.choice(os.listdir(img1_class_path)))

                if should_get_same_class:
                    img0_class_path = img1_class_path
                    while True:
                        img0_path = os.path.join(img0_class_path, random.choice(os.listdir(img0_class_path)))
                        if self._get_class_name(img0_path) == self._get_class_name(img1_path):
                            break
                else:
                    img0_class_path = os.path.join(self.root_dir, random.choice([c for c in os.listdir(self.root_dir) if c != img1_class]))
                    img0_path = os.path.join(img0_class_path, random.choice(os.listdir(img0_class_path)))

                img0_cv2 = _load_image(img0_path)
                img1_cv2 = _load_image(img1_path)

                ocr1 = do_ocr(img0_cv2)
                ocr2 = do_ocr(img1_cv2)
                ocr_similarity = ocr_check(ocr1, ocr2)

                orb_matches, total_matches = orb(img0_cv2, img1_cv2)
                orb_similarity = orb_matches / total_matches if total_matches > 0 else 0

                nh1 = SiameseNetworkDataset.get_neural_hash(self.nh_session, self.nh_seed, img0_cv2)
                nh2 = SiameseNetworkDataset.get_neural_hash(self.nh_session, self.nh_seed, img1_cv2)
                nh_similarity = SiameseNetworkDataset.calculate_hash_similarity(nh1, nh2)

                print(f"OCR Similarity: {ocr_similarity}")
                print(f"ORB Similarity: {orb_similarity}")
                print(f"NH Similarity: {nh_similarity}")

                combined_score = calculate_similarity_scores(ocr_similarity, orb_similarity, nh_similarity)
                feature_vector = np.array([ocr_similarity, orb_similarity, nh_similarity, combined_score])

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                img0_pil = Image.open(img0_path).convert('RGB')
                img1_pil = Image.open(img1_path).convert('RGB')

                if self.transform is not None:
                    img0 = self.transform(img0_pil).unsqueeze(0).to(device)
                    img1 = self.transform(img1_pil).unsqueeze(0).to(device)

                return img0, img1, feature_vector

            except (PIL.UnidentifiedImageError, FileNotFoundError, OSError) as e:
                print(f"Skipping invalid image: {e}")
                continue
        
    def _get_class_name(self, path):
        # Extract the filename without extension
        filename = os.path.splitext(os.path.basename(path))[0]
        # Split the filename by hyphen and return the first part as class name
        return filename.split('-')[0]
    
    def __len__(self):  
        return len(self.image_paths)