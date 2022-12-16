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

test_batch_size = 8
batch_size = 8
start_epoch = 0
epochs = 10
lr = 0.01
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


def get_model_name(s, lens):
    return os.path.join(os.getcwd(), "saves/U-Net-" + str(s + lens) + "th.model")
def get_optimizer_name(s, lens):
    return os.path.join(os.getcwd(), "saves/U-Net-" + str(s + lens) + "th.opt")
def get_scheduler_name(s, lens):
    return os.path.join(os.getcwd(), "saves/U-Net-" + str(s + lens) + "th.sche")

def get_loss_name_pdf(s, lens):
    return os.path.join(os.getcwd(), "saves/U-Net-" + str(s + lens) + "th-loss.pdf")


def get_loss_name_npy():
    return os.path.join(os.getcwd(), "saves/U-Net-loss.npy")


def get_train_iou_npy():
    return os.path.join(os.getcwd(), "saves/U-Net-train-iou.npy")

def get_val_iou_npy():
    return os.path.join(os.getcwd(), "saves/U-Net-val-iou.npy")

def get_iou_pdf(s, lens):
    return os.path.join(os.getcwd(), "saves/U-Net-" + str(s + lens) + "th-iou.pdf")




def cal_iou(data_loader, model):
    acc_ratios = []
    class_values = list(range(9))  # [0, 1, 2. ... 9]
    totcnt = [0] * 9
    tmp_tot = [0] * 9
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        Y_pred = model(X)
        Y_pred = torch.argmax(Y_pred, dim=1)
        for class_value in class_values:
            intersection = torch.logical_and(Y == class_value, Y_pred == class_value).sum(dim=(1, 2))
            union = torch.logical_or(Y == class_value, Y_pred == class_value).sum(dim=(1, 2))
            for i in range(Y.shape[0]):
                if union[i] > 0:
                    tmp_tot[class_value] += intersection[i] / union[i]
                    totcnt[class_value] += 1
    for i in class_values:
        acc_ratios.append((tmp_tot[i] / totcnt[i]).item())
    return acc_ratios


train_dir = os.path.join(os.getcwd(), "new_data", "train")
image_dir = os.path.join(train_dir, "images")
mask_dir = os.path.join(train_dir, "masks")
train_dataset = MyDataset(image_dir=image_dir, mask_dir=mask_dir, train_dir=train_dir)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size)

val_dir = os.path.join(os.getcwd(), "new_data", "val")
image_dir = os.path.join(val_dir, "images")
mask_dir = os.path.join(val_dir, "masks")
test_dataset = MyDataset(image_dir=image_dir, mask_dir=mask_dir, train_dir=val_dir)
test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# 加载上次的模型与loss数据
model = UNet(num_classes=9).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)
epoch_losses = []
train_iou_arr = []
val_iou_arr = []
if start_epoch > 0:
    model.load_state_dict(torch.load(get_model_name(start_epoch, 0)))
    optimizer.load_state_dict(torch.load(get_optimizer_name(start_epoch, 0)))
    scheduler.load_state_dict(torch.load(get_scheduler_name(start_epoch, 0)))
    if os.path.exists(get_loss_name_npy()):
        epoch_losses = np.load(get_loss_name_npy()).tolist()
    if os.path.exists(get_train_iou_npy()):
        train_iou_arr = np.load(get_train_iou_npy()).tolist()
    if os.path.exists(get_val_iou_npy()):
        val_iou_arr = np.load(get_val_iou_npy()).tolist()
epoch_losses = epoch_losses[:start_epoch]
train_iou_arr = train_iou_arr[:start_epoch]
val_iou_arr = val_iou_arr[:start_epoch]


# 开始训练


for epoch in range(epochs):
    epoch += 1 # start from 1
    print("on iteration:" + str(start_epoch + epoch))
    start_time = time.time()
    epoch_loss = 0
    for X, Y in train_data_loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_losses.append(epoch_loss / len(train_data_loader))

    # 每次迭代计算训练集和验证集的iou
    cur_train_iou = cal_iou(train_data_loader, model)
    cur_val_iou = cal_iou(test_data_loader, model)
    train_iou_arr.append(cur_train_iou)
    val_iou_arr.append(cur_val_iou)
    # 保存模型与数据
    if epoch % 10 == 0:
        torch.save(model.state_dict(), get_model_name(start_epoch, epoch))
        torch.save(optimizer.state_dict(), get_optimizer_name(start_epoch, epoch))
        torch.save(scheduler.state_dict(), get_scheduler_name(start_epoch, epoch))

    np.save(get_loss_name_npy(), epoch_losses)
    np.save(get_train_iou_npy(), train_iou_arr)
    np.save(get_val_iou_npy(), val_iou_arr)

    scheduler.step()
    end_time = time.time()
    print("curt: " + str(end_time - start_time) + "/s")

# 打印并保存loss图
fig, axe = plt.subplots(figsize=(10, 10))
x_axe = list(range(start_epoch + epochs))
x_axe = [i + 1 for i in x_axe]
axe.plot(x_axe, epoch_losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(get_loss_name_pdf(start_epoch, epochs))

# 打印并保存iou图
fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
plt.xlabel('epoch')
plt.ylabel('iou')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
for row in range(3):
    for col in range(3):
        axes2[row, col].plot(x_axe, [x[row * 3 + col] for x in train_iou_arr])
        axes2[row, col].plot(x_axe, [x[row * 3 + col] for x in val_iou_arr])
        axes2[row, col].set_title("IOU of class " + str(row * 3 + col))
        axes2[row, col].legend(['train', 'val'])
plt.savefig(get_iou_pdf(start_epoch, epochs))
