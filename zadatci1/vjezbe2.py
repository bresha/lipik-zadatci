import cv2
import numpy as np


image = cv2.imread('kamera1.jpg', cv2.IMREAD_COLOR)

image[10:50:5, 40:60:5, :] = [0, 0, 255]


slika = image[:, :, ::-1]
print(image[10:50, 40:60, :].shape)

cv2.imshow('Slika', image)
cv2.imshow('slika2', slika)

cv2.waitKey(0)
cv2.destroyAllWindows()
