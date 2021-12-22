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

canny = cv2.Canny(gray, 150, 220)

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
lines = cv2.HoughLines(img_mask, 1, np.pi/180, 150)


reshaped_lines = lines.reshape(lines.shape[0], lines.shape[2])

least_theta_index = reshaped_lines[:, 1].argmin()
highest_theta_index = reshaped_lines[:, 1].argmax()

filtered_lines = list()
filtered_lines.append(reshaped_lines[least_theta_index])
filtered_lines.append(reshaped_lines[highest_theta_index])

for line in filtered_lines:
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

    cv2.line(img, pt1, pt2, (0,0,255), 5)

cv2.imshow('Original', img)
cv2.imshow('Canny', canny)

cv2.waitKey()
cv2.destroyAllWindows()
