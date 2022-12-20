# 将data文件夹划分成训练集和验证集，比例为4：1
# 假设images和mask的文件都是一一对应的
# 不要再跑一遍
import os
import random
from shutil import copy2

pwd = os.path.join(os.getcwd(), "data")  # /data
image_path = os.path.join(pwd, "images")
masks_path = os.path.join(pwd, "masks")
new_path = os.path.join(os.getcwd(), "new_data")  # ./
split_rate = 0.8
image_names = os.listdir(image_path)
split_names = ['train', 'val']
to_map = {}

# 创建目录
if os.path.isdir(new_path):
    pass
else:
    os.makedirs(new_path)

csv_src = os.path.join(os.getcwd(), "data", "labels_class_dict.csv")
# 创建子目录
for split_name in split_names:
    split_path = new_path + "/" + split_name
    p1 = split_path + "/" + "images"
    p2 = split_path + "/" + "masks"
    if os.path.isdir(p1):
        for i in os.listdir(p1):
            path = p1 + "/" + i
            os.remove(path)
    else:
        os.makedirs(p1)
    if os.path.isdir(p2):
        for i in os.listdir(p2):
            path = p2 + "/" + i
            os.remove(path)
    else:
        os.makedirs(p2)
    csv_to = os.path.join(split_path, "labels_class_dict.csv")
    copy2(csv_src, csv_to)


def split(full_list, shuffle=False, ratio=0.5):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


sublist1_, sublist2_ = split(image_names, True, split_rate)

for name in sublist1_:
    image_src_path = os.path.join(image_path, name)
    mask_src_path = os.path.join(masks_path, name.replace('jpg', 'png'))
    image_dst_path = os.path.join(new_path, split_names[0], "images")
    mask_dst_path = os.path.join(new_path, split_names[0], "masks")
    copy2(image_src_path, image_dst_path)
    copy2(mask_src_path, mask_dst_path)

for name in sublist2_:
    image_src_path = os.path.join(image_path, name)
    mask_src_path = os.path.join(masks_path, name.replace('jpg', 'png'))
    image_dst_path = os.path.join(new_path, split_names[1], "images")
    mask_dst_path = os.path.join(new_path, split_names[1], "masks")
    copy2(image_src_path, image_dst_path)
    copy2(mask_src_path, mask_dst_path)
