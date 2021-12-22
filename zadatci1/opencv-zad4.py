'''
Ponovite prethodni zadatak, ali boju izmijenite koristeći HSV format
'''

import cv2

img = cv2.imread('kamera2_1.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv[273: 282, 143 : 154] = [24, 255, 255]

back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('Original', img)
cv2.imshow('Rect', back)

cv2.waitKey(0)
cv2.destroyAllWindows()
