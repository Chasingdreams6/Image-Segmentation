import itertools

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
from sklearn.metrics import confusion_matrix
import time
from torchsummary import summary

test_batch_size = 4
batch_size = 4
start_epoch = 0
epochs = 60
insight_epoch = 60
lr = 0.00025
milestones = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400]
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


def get_mean_iou_npy():
    return os.path.join(os.getcwd(), "saves/U-Net-mean-iou.npy")


def get_iou_pdf(s, lens):
    return os.path.join(os.getcwd(), "saves/U-Net-" + str(s + lens) + "th-iou.pdf")


def get_miou_pdf(s, lens):
    return os.path.join(os.getcwd(), "saves/U-Net-" + str(s + lens) + "th-miou.pdf")


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


def cal_mean_iou(s, t, mapp, val_iou):
    mean_iou = 0
    for i in range(s, t):  # last ten
        cur = 0
        for row in range(3):
            for col in range(3):
                cur += val_iou[i][row * 3 + col] * mapp[row * 3 + col]

        mean_iou += cur
    mean_iou /= (t - s)
    return mean_iou


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

# 总点数统计
mapp = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
for X, Y in test_data_loader:
    mask = Y
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                mapp[Y[i][j][k].item()] += 1

sum_point = 0
for i in range(9):
    sum_point += mapp[i]

for i in range(9):
    mapp[i] /= sum_point

# 加载上次的模型与loss数据
model = UNet(num_classes=9).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
epoch_losses = []
train_iou_arr = []
val_iou_arr = []
mean_iou_arr = []
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
    if os.path.exists(get_mean_iou_npy()):
        mean_iou_arr = np.load(get_mean_iou_npy()).tolist()
epoch_losses = epoch_losses[:start_epoch]
train_iou_arr = train_iou_arr[:start_epoch]
val_iou_arr = val_iou_arr[:start_epoch]
mean_iou_arr = mean_iou_arr[:start_epoch]
# summary(model, (3, 256, 256))

Y_preds = []
Y_s = []
# 开始训练
for epoch in range(epochs):
    epoch += 1  # start from 1
    print("on iteration:" + str(start_epoch + epoch))
    start_time = time.time()
    epoch_loss = 0
    for X, Y in train_data_loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)  # 8 * 9 * 256 * 256
        loss = criterion(Y_pred, Y)  # 8 * 256 * 256
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_losses.append(epoch_loss / len(train_data_loader))
    print("loss: " + str(epoch_loss / len(train_data_loader)))
    # 每次迭代计算训练集和验证集的iou
    cur_train_iou = cal_iou(train_data_loader, model)
    cur_val_iou = cal_iou(test_data_loader, model)
    train_iou_arr.append(cur_train_iou)
    val_iou_arr.append(cur_val_iou)
    cur_mean_iou = cal_mean_iou(start_epoch + epoch - 1, start_epoch + epoch, mapp, val_iou_arr)
    print("cur mean iou: " + str(cur_mean_iou))
    mean_iou_arr.append(cur_mean_iou)
    # 保存模型与数据
    if (start_epoch + epoch) % 20 == 0:
        torch.save(model.state_dict(), get_model_name(start_epoch, epoch))
        torch.save(optimizer.state_dict(), get_optimizer_name(start_epoch, epoch))
        torch.save(scheduler.state_dict(), get_scheduler_name(start_epoch, epoch))

    np.save(get_loss_name_npy(), epoch_losses)
    np.save(get_train_iou_npy(), train_iou_arr)
    np.save(get_val_iou_npy(), val_iou_arr)
    np.save(get_mean_iou_npy(), mean_iou_arr)

    # # 保存混淆矩阵数据
    # if insight_epoch == epoch + start_epoch:
    #     for X, Y in test_data_loader:
    #         X, Y = X.to(device), Y.to(device)
    #         Y_pred = model(X)  # 8 * 9 * 256 * 256
    #         Y_preds = Y_preds + torch.argmax(Y_pred, dim=1).reshape(-1).tolist()
    #         Y_s = Y_s + Y.reshape(-1).tolist()

    scheduler.step()
    end_time = time.time()
    print("curt: " + str(end_time - start_time) + "/s")

# 保存最后的混淆矩阵数据
for X, Y in test_data_loader:
    X, Y = X.to(device), Y.to(device)
    Y_pred = model(X)  # 8 * 9 * 256 * 256
    Y_preds = Y_preds + torch.argmax(Y_pred, dim=1).reshape(-1).tolist()
    Y_s = Y_s + Y.reshape(-1).tolist()

np.save("Y_preds.npy", Y_preds)
np.save("Y_s.npy", Y_s)


# 输出混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred, labels=labels)  # 可将'1'等替换成自己的类别，如'cat'。
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    # plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = float('{:.2f}'.format(cm[i, j])) if normalize else int(cm[i, j])
        plt.text(i, j, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("saves/confusion.svg", bbox_inches='tight')


plot_confusion_matrix(Y_s, Y_preds, [0, 1, 2, 3, 4, 5, 6, 7, 8], normalize=True)

# 打印并保存loss图
fig, axe = plt.subplots(figsize=(10, 10))
x_axe = list(range(start_epoch + epochs))
x_axe = [i + 1 for i in x_axe]
axe.plot(x_axe, epoch_losses)
plt.xlabel('epoch')
plt.ylabel('loss')
axe.set_title('loss lr=' + str(lr))
plt.savefig(get_loss_name_pdf(start_epoch, epochs))

# 打印并保存miou图
fig, axe = plt.subplots(figsize=(10, 10))
axe.plot(x_axe, mean_iou_arr)
plt.xlabel('epoch')
plt.ylabel('mean iou')
axe.set_title('mean iou')
plt.savefig(get_miou_pdf(start_epoch, epochs))

# 打印并保存iou图
fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
for row in range(3):
    for col in range(3):
        axes2[row, col].plot(x_axe, [x[row * 3 + col] for x in train_iou_arr])
        axes2[row, col].plot(x_axe, [x[row * 3 + col] for x in val_iou_arr])
        axes2[row, col].set_title("IOU of class " + str(row * 3 + col))
        axes2[row, col].legend(['train', 'val'])
        axes2[row, col].set_xlabel('epoch')
        axes2[row, col].set_ylabel('iou')
        axes2[row, col].set_yticks([0, .2, .4, .6, .8, 1])
plt.savefig(get_iou_pdf(start_epoch, epochs))

# # MIOU计算
#
# mean_iou = cal_mean_iou(start_epoch + epochs - 10, start_epoch + epochs, mapp, val_iou_arr)
# print("mean_iou: " + str(mean_iou))

# 输出预测的图片

# 把前8个图片显示出来
X, Y = next(iter(test_data_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model(X)
Y_pred = torch.argmax(Y_pred, dim=1)
inverse_transform = transforms.Compose([
    transforms.Normalize((-0.482 / 0.234, -0.490 / 0.234, -0.474 / 0.254), (1 / 0.234, 1 / 0.234, 1 / 0.254))
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
    axes[i, 2].set_title("Label Class_pred")

plt.savefig("saves/prediction.pdf")
