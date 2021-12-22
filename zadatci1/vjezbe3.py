import cv2

img_gray = cv2.imread('kamera1.jpg', 0)

ret, tresh = cv2.threshold(img_gray, 75, 255, cv2.THRESH_BINARY)

cv2.imshow('Origigi', img_gray)

cv2.imshow('Binary', tresh)

cv2.waitKey(0)
cv2.destroyAllWindows()