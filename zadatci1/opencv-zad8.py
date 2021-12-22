'''
Učitajte sliku “kamera1.png”, te detektirajte rubove pomoću Canny detektora rubova. Koje su
razlike u odnosu na sliku iz prethodnog zadatka u kojem smo koristili Sobel operator?
'''

import cv2

img = cv2.imread('kamera1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray, 150, 220)

cv2.imshow('Original', img)
cv2.imshow('Canny', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
