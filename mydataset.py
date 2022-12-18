import os

import cv2
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
from torchvision.transforms import InterpolationMode

from tqdm.notebook import tqdm
from load_data_example import one_hot_encode, reverse_one_hot


class MyDataset(Dataset):

    def __init__(self, image_dir, mask_dir, train_dir):
        self.Size = (256, 256)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_fns = os.listdir(image_dir)
        self.mask_fns = os.listdir(mask_dir)
        self.class_dict = pd.read_csv(os.path.join(train_dir, 'labels_class_dict.csv'))
        # Get class names
        self.class_names = self.class_dict['class_names'].tolist()
        # Get class RGB values
        self.class_rgb_values = self.class_dict[['r', 'g', 'b']].values.tolist()
        self.image_preprocessed = []
        self.mask_preprocessed = []

        for index in range(0, len(self.image_fns)):
            image_file_name = self.image_fns[index]
            image_path = os.path.join(self.image_dir, image_file_name)
            mask_file_name = self.mask_fns[index]
            mask_path = os.path.join(self.mask_dir, mask_file_name)
            #image = Image.open(image_path).convert('RGB')
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image = np.array(image)
            image = self.transform(image)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            #mask = Image.open(mask_path).convert('RGB')
            mask = np.array(mask)
            mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
            mask = reverse_one_hot(mask)
            mask = torch.Tensor(mask).long()
            mask = mask.unsqueeze(0)  # 升一维
            mask_transform = transforms.Resize(size=self.Size, interpolation=InterpolationMode.NEAREST)
            mask = mask_transform(mask)
            mask = mask.squeeze(0)  # 还原维度
            self.image_preprocessed.append(image)
            self.mask_preprocessed.append(mask)

    def __len__(self):
        return len(self.image_fns)

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.482, 0.490, 0.474), std=(0.234, 0.234, 0.254)),
            transforms.Resize(size=self.Size, interpolation=InterpolationMode.NEAREST),
            transforms.GaussianBlur(kernel_size=7)
        ])
        return transform_ops(image)

    def __getitem__(self, index):
        return self.image_preprocessed[index], self.mask_preprocessed[index]
