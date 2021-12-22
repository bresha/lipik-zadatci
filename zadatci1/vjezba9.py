import cv2
import numpy as np

img = np.zeros((500,500,3))

c_center = (100,100)
c_radius = 100
c_color = [0,0,255]

cv2.circle(img, c_center, c_radius, c_color)

cv2.imshow('krug', img)
cv2.waitKey()
cv2.destroyAllWindows()
