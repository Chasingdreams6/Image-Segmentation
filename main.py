import os
from mydataset import MyDataset

train_dir = os.path.join(os.getcwd(), "new_data", "train")
image_dir = os.path.join(train_dir, "images")
mask_dir = os.path.join(train_dir, "masks")

dataset = MyDataset(image_dir=image_dir, mask_dir=mask_dir, train_dir=train_dir)
print(len(dataset))