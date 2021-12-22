'''
Pomoću OpenCV-a učitajte sliku koja je snimljena s kamere automobila pod nazivom
“kamera2.jpg”. Izvršite binarizaciju slike gdje će prag binarizacije imati vrijednost aritmetičke
sredine svih piksela slike
'''

import cv2

img_gray = cv2.imread('kamera2_1.jpg', 0)

ret, thresh = cv2.threshold(img_gray, img_gray.mean(), img_gray.max(), cv2.THRESH_BINARY)

cv2.imshow('Binary', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()