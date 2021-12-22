import cv2

img = cv2.imread('kamera2.jpg', 0)

canny = cv2.Canny(img, 100, 200)

cv2.imshow('Original', img)
cv2.imshow('Canny', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()