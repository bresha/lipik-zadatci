import cv2
import numpy as np

img = cv2.imread('kamera1.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lo = np.array([75, 80, 80])
hi = np.array([88, 255, 255])

mask = cv2.inRange(img_hsv, lo, hi)

masked_img = img * (mask[:,:,None] // 255)

print(masked_img.shape)
cv2.imshow('Orig', img)
cv2.imshow('Masked', masked_img)

cv2.waitKey(0)

cv2.destroyAllWindows()

