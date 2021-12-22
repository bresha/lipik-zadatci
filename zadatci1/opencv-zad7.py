'''
Učitajte sliku “kamera1.png”, te detektirajte rubove pomoću Sobel operatora. Prikažite sliku koja
sadrži vrijednosti gradijenata u x i y smjeru.
'''

import cv2

img = cv2.imread('kamera1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gauss = cv2.GaussianBlur(gray, (3, 3), 0)

sobel = cv2.Sobel(gauss, cv2.CV_64F, 1, 1, ksize=5)

sobel_abs = cv2.convertScaleAbs(sobel)

cv2.imshow('Original', img)
cv2.imshow("Sobel", sobel_abs)

cv2.waitKey(0)
cv2.destroyAllWindows()
