import cv2
import numpy as np

im = cv2.imread('kamera4.png', 0)
mask = np.zeros_like(im)
height, width = im.shape

x1 = width / 2
y1 = 0
x2 = width
y2 = 0
x3 = width
y3 = height
x4 = width / 2
y4 = height

rectangle_points = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]], dtype=np.int32)

mask = cv2.fillPoly(mask, rectangle_points, 255)
masked_image = cv2.bitwise_and(im, mask)

cv2.imshow('Orig', im)
cv2.imshow('mask', mask)
cv2.imshow('maskirana slika', masked_image)
cv2.waitKey()
cv2.destroyAllWindows()
