import os
from mydataset import MyDataset
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

epoch_losses = {2: [], 4: [], 8: [], 16: [], 32: []}
train_iou_arr = {2: [], 4: [], 8: [], 16: [], 32: []}
val_iou_arr = {2: [], 4: [], 8: [], 16: [], 32: []}
start_epoch = 100
array = [2, 4, 8, 16, 32]
str_array = ["batch=2", "batch=4", "batch=8", "batch=16", "batch=32"]


def get_loss_name_npy(bs):
    return os.path.join(os.getcwd(), "ious", "b" + str(bs) + "-U-Net-loss.npy")


def get_train_iou_npy(bs):
    return os.path.join(os.getcwd(), "ious", "b" + str(bs) + "-U-Net-train-iou.npy")


def get_val_iou_npy(bs):
    return os.path.join(os.getcwd(), "ious", "b" + str(bs) + "-U-Net-val-iou.npy")


for batch_size in array:
    if start_epoch > 0:
        epoch_losses[batch_size] = np.load(get_loss_name_npy(batch_size)).tolist()
        train_iou_arr[batch_size] = np.load(get_train_iou_npy(batch_size)).tolist()
        val_iou_arr[batch_size] = np.load(get_val_iou_npy(batch_size)).tolist()

# 打印并保存loss图
fig, axe = plt.subplots(figsize=(10, 10))
x_axe = list(range(start_epoch))
x_axe = [i + 1 for i in x_axe]
for batch_size in array:
    axe.plot(x_axe, epoch_losses[batch_size])
plt.legend(str_array)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("loss-batch.pdf")

# 打印并保存iou图
fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
for row in range(3):
    for col in range(3):
        for batch_size in array:
            # axes2[row, col].plot(x_axe, [x[row * 3 + col] for x in train_iou_arr])
            axes2[row, col].plot(x_axe, [x[row * 3 + col] for x in val_iou_arr[batch_size]])
        axes2[row, col].set_title("IOU of class " + str(row * 3 + col))
        axes2[row, col].legend(str_array)
        axes2[row, col].set_xlabel('epoch')
        axes2[row, col].set_ylabel('iou')
        axes2[row, col].set_yticks([0, .2, .4, .6, .8, 1])
plt.savefig("val-iou-batch.pdf")

# 打印并保存iou图
fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
for row in range(3):
    for col in range(3):
        for batch_size in array:
            axes2[row, col].plot(x_axe, [x[row * 3 + col] for x in train_iou_arr[batch_size]])
            # axes2[row, col].plot(x_axe, [x[row * 3 + col] for x in val_iou_arr[batch_size]])
        axes2[row, col].set_title("IOU of class " + str(row * 3 + col))
        axes2[row, col].legend(str_array)
        axes2[row, col].set_xlabel('epoch')
        axes2[row, col].set_ylabel('iou')
        axes2[row, col].set_yticks([0, .2, .4, .6, .8, 1])
plt.savefig("train-iou-batch.pdf")
