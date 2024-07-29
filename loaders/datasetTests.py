import numpy as np
import random
import torch
import PIL.ImageOps
from torch.utils.data import Dataset
from params.config import Config
from PIL import Image
import os

class TestSiameseNetworkDataset(Dataset):

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
        img0_path = self.image_paths[self.random_indexes[int(index / Config.train_batch_size)]]
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            while True:
                img1_path = random.choice(self.image_paths)
                if self._get_class_name(img0_path) == self._get_class_name(img1_path):
                    break
        else:
            while True:
                img1_path = random.choice(self.image_paths)
                if self._get_class_name(img0_path) != self._get_class_name(img1_path):
                    break

        img0 = self._load_image(img0_path)
        img1 = self._load_image(img1_path)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        class_name_1 = self._get_class_name(img0_path)
        class_name_2 = self._get_class_name(img1_path)
        label = torch.tensor(int(class_name_1 != class_name_2), dtype=torch.float32)

        return img0, img1, label
    
    def _get_class_name(self, path):
        # Extract the filename without extension
        filename = os.path.splitext(os.path.basename(path))[0]
        # Split the filename by hyphen and return the first part as class name
        return filename.split('-')[0]
    
    def _load_image(self, path):
        if path in self.image_cache:
            return self.image_cache[path]

        img = Image.open(path)

        if self.should_invert:
            img = PIL.ImageOps.invert(img.convert('RGB'))

        # Cache the loaded image
        self.image_cache[path] = img
        return img

    def __len__(self):
        return len(self.image_paths)
