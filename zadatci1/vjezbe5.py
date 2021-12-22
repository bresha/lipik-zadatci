import cv2

img = cv2.imread('kamera1.jpg')
cv2.imshow('Original', img)

blur = cv2.blur(img, (5,5))
cv2.imshow('Blur', blur)

gauss = cv2.GaussianBlur(img, (5,5), 0)
cv2.imshow('Gauss', gauss)


cv2.waitKey(0)

cv2.destroyAllWindows()