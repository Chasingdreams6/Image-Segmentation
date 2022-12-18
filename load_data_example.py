import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# helper function for data visualization


def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    # print(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        # print(name)
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)

    plt.savefig('visualization.png')

# Perform one hot encoding on label


def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour) # np.equal实现把label image每个像素的RGB值与某个class的RGB值进行比对，变成RGB bool值
        class_map = np.all(equality, axis=-1) # np.all 把RGB bool值，变成一个bool值，即实现某个class 的label mask。使用for循环，生成所有class的label mask
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1) # np.stack实现所有class的label mask的堆叠。最终depth size 为num_classes的数量

    return semantic_map

# Perform reverse one-hot-encoding on labels / preds


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 

    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis=-1) # axis表示最后一个维度，即channel
    return x


if __name__ == '__main__':

    DATA_DIR = 'data'
    class_dict = pd.read_csv(os.path.join(DATA_DIR, 'labels_class_dict.csv'))
    # Get class names
    class_names = class_dict['class_names'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

    print('All dataset classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)

    image_names = os.listdir(os.path.join(DATA_DIR, 'images'))
    image_paths = [os.path.join(DATA_DIR, 'images', image_name) for image_name in image_names]
    mask_paths = [os.path.join(DATA_DIR, 'masks', image_name.replace('jpg', 'png')) for image_name in image_names]

    # example code to read one image and its corresponding mask
    example_index = 0

    # trans BGR to RGB
    image = cv2.cvtColor(cv2.imread(image_paths[0]), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(cv2.imread(mask_paths[0]), cv2.COLOR_BGR2RGB)

    #visualize(original_image=image, mask=mask)
    # visualize(nam)
    # visualize(image_names, image_paths)
    mask = one_hot_encode(mask, class_rgb_values).astype('float')
    print(mask.shape)
    mask = reverse_one_hot(mask)

    print('Image shape: ', image.shape)
    print('Mask shape: ', mask)

    # for example_index in range(len(image_paths)):
    #     image = cv2.cvtColor(cv2.imread(image_paths[example_index]), cv2.COLOR_BGR2RGB)
    #     mask = cv2.cvtColor(cv2.imread(mask_paths[example_index]), cv2.COLOR_BGR2RGB)
    #     mask = one_hot_encode(mask, class_rgb_values).astype('float')
    #     print()