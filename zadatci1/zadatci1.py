'''
1. Pomoću Houghove transformacije detektirajte i nacrtajte pravce na slici “kamera3.jpeg”. Među
detektiranim pravcima moraju biti detektirane i linije voznih traka.

2. Primijetite nedostatke prethodnog zadatka. Teško je detektirati samo linije vozne trake bez
detekcije i ostalih linija na slici. Metodom maskiranja slike, pokušajte izdvojiti samo linije voznih
traka.
'''


import cv2
import numpy as np

img = cv2.imread('kamera3.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3,3), 0)

canny = cv2.Canny(blur, 150, 220)

heigth, width = gray.shape
x1 = 0
y1 = heigth
x2 = width
y2 = heigth
x3 = width // 2
y3 = int(2/5 * heigth)


triangle_points = np.array([[(x1, y1), (x2, y2), (x3, y3)]], dtype=np.int32)

mask = np.zeros_like(gray)
mask = cv2.fillPoly(mask, triangle_points, 255)

img_mask = cv2.bitwise_and(canny, mask)
lines = cv2.HoughLinesP(img_mask, 1, np.pi/180, 50, None, 100, 20)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

cv2.imshow('Original', img)
cv2.imshow('Canny', canny)

cv2.waitKey()
cv2.destroyAllWindows()
