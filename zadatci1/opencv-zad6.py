'''
Učitajte sliku “crveni_semafor.jpg”, te filtrirajte crvenu boju sa slike. Rezultat filtriranja mora biti
slika koja sadrži samo crvenu strjelicu sa semafora.
'''


import cv2
import numpy as np

img = cv2.imread('crveni_semafor.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lo = np.array([165, 80, 80])
hi = np.array([179, 255, 255])


mask = cv2.inRange(hsv, lo, hi)

filter_img_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
filter_img_bgr = cv2.cvtColor(filter_img_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('Original', img)
cv2.imshow("Filter Image", filter_img_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()
