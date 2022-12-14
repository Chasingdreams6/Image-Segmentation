import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm
from load_data_example import one_hot_encode, reverse_one_hot

class MyDataset(Dataset):

    def __init__(self, image_dir, mask_dir, train_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_fns = os.listdir(image_dir)
        self.mask_fns = os.listdir(mask_dir)
        self.class_dict = pd.read_csv(os.path.join(train_dir, 'labels_class_dict.csv'))
        # Get class names
        self.class_names = self.class_dict['class_names'].tolist()
        # Get class RGB values
        self.class_rgb_values = self.class_dict[['r', 'g', 'b']].values.tolist()

    def __len__(self):
        return len(self.image_fns)

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

    def __getitem__(self, index):
        image_file_name = self.image_fns[index]
        image_path = os.path.join(self.image_dir, image_file_name)
        mask_file_name = self.mask_fns[index]
        mask_path = os.path.join(self.mask_dir, mask_file_name)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = self.transform(image)
        mask = Image.open(mask_path).convert('RGB')
        mask = np.array(mask)
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        mask = reverse_one_hot(mask)
        return image, mask
