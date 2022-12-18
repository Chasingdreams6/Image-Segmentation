from load_data_example import one_hot_encode, reverse_one_hot
from mydataset import MyDataset
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
from unet import UNet
import time

val_dir = os.path.join(os.getcwd(), "new_data", "val")
image_dir = os.path.join(val_dir, "images")
mask_dir = os.path.join(val_dir, "masks")
test_batch_size = 8

train_dir = os.path.join(os.getcwd(), "new_data", "train")
image_dir = os.path.join(train_dir, "images")
mask_dir = os.path.join(train_dir, "masks")
class_dict = pd.read_csv(os.path.join(train_dir, 'labels_class_dict.csv'))

dataset = MyDataset(image_dir=image_dir, mask_dir=mask_dir, train_dir=train_dir)
data_loader = DataLoader(dataset, batch_size=test_batch_size)

# Get class names
class_names = class_dict['class_names'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
res = []
mapp = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
for X, Y in data_loader:
    mask = Y
    print(mask.shape)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                mapp[Y[i][j][k].item()] += 1
print(mapp)
