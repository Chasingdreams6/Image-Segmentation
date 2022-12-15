from mydataset import MyDataset
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
from unet import UNet
import time

train_dir = os.path.join(os.getcwd(), "new_data", "train")
image_dir = os.path.join(train_dir, "images")
mask_dir = os.path.join(train_dir, "masks")

dataset = MyDataset(image_dir=image_dir, mask_dir=mask_dir, train_dir=train_dir)
print(len(dataset))

# for i in range(0, 25):
#     x, y = dataset[i]
#     print(x.shape)
#     print(y.shape)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)

batch_size = 8
epochs = 5
lr = 0.01
data_loader = DataLoader(dataset, batch_size=batch_size)
model = UNet(num_classes=9).to(device)
model.load_state_dict(torch.load("U-Net.5th"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
step_losses = []
epoch_losses = []
for epoch in range(epochs):
    print("epoch:" + str(epoch))
    epoch_loss = 0
    for X, Y in tqdm(data_loader, total=len(data_loader)):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
    epoch_losses.append(epoch_loss/len(data_loader))
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(step_losses)
axes[1].plot(epoch_losses)
plt.savefig("loss100.pdf")
model_name = "U-Net.5th"
torch.save(model.state_dict(), model_name)

val_dir = os.path.join(os.getcwd(), "new_data", "val")
image_dir = os.path.join(val_dir, "images")
mask_dir = os.path.join(val_dir, "masks")
test_batch_size = 8
dataset = MyDataset(image_dir=image_dir, mask_dir=mask_dir, train_dir=val_dir)
data_loader = DataLoader(dataset, batch_size=test_batch_size)
X, Y = next(iter(data_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model(X)
print(Y_pred.shape)
Y_pred = torch.argmax(Y_pred, dim=1)
print(Y_pred.shape)
inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])
fig, axes = plt.subplots(test_batch_size, 3, figsize=(3 * 5, test_batch_size * 5))

for i in range(test_batch_size):
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()

    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")

plt.savefig("100epoch.pdf")
