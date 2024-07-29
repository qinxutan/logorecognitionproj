import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

import tensorflow as tf
import cv2
from PIL import Image
import os

import paddle
from paddleocr import PaddleOCR
import textdistance
import logging
import sys
import time
import numpy as np
import re
import csv
import onnxruntime
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from models.SNbetternet import SiameseNetwork
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from params.config import Config


from PIL import Image

from loaders.datasetTests import TestSiameseNetworkDataset
from params.config import Config
import math

from misc.misc import Utils

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

# Function to load and preprocess an image
def get_img(path):
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

        if new_img is None:
            return None
        if len(new_img.shape) == 2:
                return new_img
        else:
            gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            return gray_img
    
    except Exception as e:
        return None

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
        min_match_length = len(textdistance.lcsseq(ocr1_clean, ocr2_clean))
        match_str_ratio = min_match_length / min_ocr_length
        
        if text_sim < 0.2 or match_str_ratio < 0.6 or (text_sim < 0.3 and len(ocr1_clean) < 10 and len(ocr2_clean) < 10):
            check_ocr = False
    else:
        check_ocr = False
    return text_sim if check_ocr else 0

class Tester:
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
            return "NULL"
        
    def calculate_hash_similarity(hash1, hash2):
        if hash1 == "NULL" or hash2 == "NULL":
            return 0.0

        distance = sum(1 for x, y in zip(hash1, hash2) if x != y)
        similarity = 1.0 - (distance / len(hash1))

        return similarity
    
    def calculate_similarity_scores(ocr_sim, orb_sim, nh_sim):
        # Define weight for each feature
        weights = {'ocr': Config.ocr_weight, 'orb': Config.orb_weight, 'nh': Config.nh_weight}

        # Adjusting threshold values
        ocr_threshold = Config.ocr_threshold
        orb_threshold = Config.orb_threshold
        nh_threshold = Config.nh_threshold

        # Weighted score calculation
        score = (weights['ocr'] * (ocr_sim if ocr_sim > ocr_threshold else 0) +
                weights['orb'] * (orb_sim if orb_sim > orb_threshold else 0) +
                weights['nh'] * (nh_sim if nh_sim > nh_threshold else 0))
        
        return score

    
    @staticmethod
    def test(): 
        print("Testing process initialized...")
        print("dataset: ", Config.testing_dir)

        nh_session, nh_seed = Tester.load_neural_hash_model()
        if nh_session is None or nh_seed is None:
            print("Failed to load neural hash model and seed. Exiting.")
            return

        query_dir = "/Users/qinxutan/Documents/htxinternship/logorecognition/logos_full/test/query"
        reference_dir = "/Users/qinxutan/Documents/htxinternship/logorecognition/logos_full/test/reference"

        query_images = []
        for root, dirs, files in os.walk(query_dir):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    query_images.append(os.path.join(root, file))

        reference_images = []
        for root, dirs, files in os.walk(reference_dir):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    reference_images.append(os.path.join(root, file))
        
        
        net = SiameseNetwork(lastLayer=True, pretrained=False)
        state_dict = torch.load(Config.best_model_path)
        net.load_state_dict(state_dict)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        net.eval()

        transform = transforms.Compose([
            transforms.Resize((Config.im_w, Config.im_h)),
            transforms.ToTensor()
        ])

        results = []
        no_match_images = []
        false_positives_list = []

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        y_true = []
        y_pred = []

        for query_img_name in query_images:
            queries_completed += 1
            print(f"Processing query image {queries_completed} of {total_queries}...")

            query_img_path = os.path.join(query_dir, query_img_name)
            if not verify_image(query_img_path):
                print(f"Error: Image is corrupted or cannot be opened for path {query_img_path}. Skipping...")
                continue
            query_img = get_img(query_img_path)
            query_img_ocr = do_ocr(query_img)
            query_img_hash = Tester.get_neural_hash(nh_session, nh_seed, query_img, query_img_path)
            try:
                queryimg_pil = Image.open(query_img_path).convert('RGB')
            except UnidentifiedImageError:
                print(f"Error: Unable to identify image file {reference_img_path}. Skipping...")
                continue 

            query = transform(queryimg_pil).unsqueeze(0).to(device)
            best_match_score = Config.best_match
            best_match_reference = None

            for reference_img_name in reference_images:
                reference_img_path = os.path.join(reference_dir, reference_img_name)
                if not verify_image(reference_img_path):
                    print(f"Error: Image is corrupted or cannot be opened for path {reference_img_path}. Skipping...")
                    continue
                reference_img = get_img(reference_img_path)
                reference_img_ocr = do_ocr(reference_img)
                try:
                    reference_img_hash = Tester.get_neural_hash(nh_session, nh_seed, reference_img, reference_img_path)
                except AttributeError:
                    print(f"Error: Unable to get neural hash for image {reference_img_path}. Skipping...")
                    continue

                try:
                    refimg_pil = Image.open(reference_img_path).convert('RGB')
                except UnidentifiedImageError:
                    print(f"Error: Unable to identify image file {reference_img_path}. Skipping...")
                    continue
                
                reference = transform(refimg_pil).unsqueeze(0).to(device)
                
                orb_sim, _ = orb(query_img, reference_img)
                orb_sim = orb_sim / 100.0
                ocr_sim = ocr_check(query_img_ocr, reference_img_ocr)
                nh_sim = Tester.calculate_hash_similarity(query_img_hash, reference_img_hash)

                combined_score = Tester.calculate_similarity_scores(ocr_sim, orb_sim, nh_sim)
                feature_vector = torch.tensor([ocr_sim, orb_sim, nh_sim, combined_score]).unsqueeze(0).to(device)

                distance = net(Variable(reference), Variable(query), feature_vector)

                if Config.bceLoss:
                    distance = torch.sigmoid(distance)
                    distance = distance.item()

                overall_score = (0.8 * combined_score) + (0.2 * distance)

                if isinstance(distance, torch.Tensor):
                    distance = distance.item()
                else:
                    distance = distance

                if overall_score > best_match_score:
                    best_match_score = overall_score
                    best_match_reference = reference_img_name
                    best_match_ocr_sim = ocr_sim
                    best_match_orb_sim = orb_sim
                    best_match_nh_sim = nh_sim
                    best_match_distance = distance

            query_img_dir = os.path.basename(os.path.dirname(query_img_path))
            is_true_positive = best_match_reference is not None and query_img_dir == os.path.basename(os.path.dirname(best_match_reference))

            if is_true_positive:
                true_positives += 1
                y_true.append(1)
                y_pred.append(1)

                for ref in reference_images:
                    if ref == query_img_name:
                        continue
                    ref_base_name = os.path.basename(os.path.dirname(ref))
                    if ref_base_name == query_img_dir:
                        true_positives += 1
                        y_true.append(1)
                        y_pred.append(1)
                    else: 
                        y_true.append(0)
                        y_pred.append(0)
                        true_negatives += 1

            else:
                false_positives += 1
                y_true.append(1)
                y_pred.append(0)
                false_positives_list.append(query_img_name)
                
                for ref in reference_images:
                    ref_base_name = os.path.basename(os.path.dirname(ref))
                    if ref_base_name == query_img_dir:
                        y_true.append(0)
                        y_pred.append(1)
                        false_negatives += 1
                    else:
                        y_true.append(0)
                        y_pred.append(0)
                        true_negatives += 1

                if (best_match_reference is None):
                    no_match_images.append(query_img_name)
            
            print("True Positives:", true_positives)
            print("False Positives:", false_positives)
            print("False Negatives:", false_negatives)
            print("True Negatives:", true_negatives)
            
            results.append({
                'query_image': query_img_name,
                'matched_reference_image': best_match_reference,
                'score': best_match_score
            })

            if best_match_reference:
                print(f"Query Image: {query_img_name} -> Matched Reference Image: {best_match_reference} with Score: {float(best_match_score):.4f}")
                print(f"OCR Sim: {best_match_ocr_sim}, ORB Sim: {best_match_orb_sim}, NH Sim: {best_match_nh_sim}, Siamese: {best_match_distance}")
            else:
                print(f"Query Image: {query_img_name} -> No Match Found")

        print("True Positives:", true_positives)
        print("False Positives:", false_positives)
        print("False Negatives:", false_negatives)
        print("True Negatives:", true_negatives)
        
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"Precision: {precision}")
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        with open('results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Query Image", "Matched Reference Image", "Score"])
            for result in results:
                writer.writerow([result['query_image'], result['matched_reference_image'], result['score']])

        # Print images with no match found
        print("\nQuery Images with No Match Found:")
        for img in no_match_images:
            print(img)
        
        print("\nQuery Images that are False Positives:")
        for img in false_positives_list:
            print(img)

if __name__ == "__main__":
    Tester.test()