import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

pedestrian_lables = np.ones((ped_dataset.shape[0]))
no_pedestrians_lables = np.zeros((no_ped_dataset.shape[0]))

X = np.concatenate((ped_dataset, no_ped_dataset))
y = np.concatenate((pedestrian_lables, no_pedestrians_lables))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = MLPClassifier(random_state=1)
classifier.fit(X_train, y_train)

score = classifier.score(X_test, y_test)
print(score)

test_image = cv2.imread('dataset/test_img.png', cv2.IMREAD_GRAYSCALE)
image_height = test_image.shape[0]
image_width = test_image.shape[1]


max_pedestrian_probability = 0
final_top_left_bb = (0, 0)
final_bottom_right_bb = (0, 0)

for i in range(0, image_height - mean_image_height - 1, 10):
    for j in range(0, image_width - mean_image_width - 1, 10):
        roi = test_image[i: i + mean_image_height, j: j + mean_image_width]
        HOG_desc, HOG_image = hog(roi, visualize=True)
        
        HOG_desc = HOG_desc.reshape((1, -1))
        roi_probabilities = classifier.predict_proba(HOG_desc)
        pedestrian_probabitiy = roi_probabilities[0, 1]

        if pedestrian_probabitiy > max_pedestrian_probability:
            max_pedestrian_probability = pedestrian_probabitiy
            final_top_left_bb = (j, i)
            final_bottom_right_bb = (j + mean_image_width, i + mean_image_height)

detection_image = cv2.rectangle(test_image, final_top_left_bb, final_bottom_right_bb, 255, 3)

cv2.imshow('detection', detection_image)
cv2.waitKey()
cv2.destroyAllWindows()


# implement Non-maximum supression

print('done')