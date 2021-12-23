import os
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

dataset_path = 'dataset'
mean_image_width = 101
mean_image_height = 264

def load_dataset(images_folder):
    image_dataset = os.path.join(dataset_path, images_folder)

    hog_list = []

    for file in os.listdir(image_dataset):
        if file.endswith('.png'):
            image_name = os.path.join(image_dataset, file)

            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (mean_image_width, mean_image_height))

            HOG_desc, hog_image = hog(image, visualize=True)

            # fig, axes = plt.subplots(1, 2)
            # axes[0].imshow(image_ped, cmap='gray')
            # axes[1].imshow(hog_image, cmap='gray')
            # plt.show()

            hog_list.append(HOG_desc)

    hog_list = np.array(hog_list)

    return hog_list




ped_dataset = load_dataset('ped')
no_ped_dataset = load_dataset('no_ped')

print('done')