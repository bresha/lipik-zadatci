import cv2

img = cv2.imread('kamera2.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (9,9), 0)

sobel2 = cv2.Sobel(blur, cv2.CV_64F, 1, 1, ksize=5)

sobel_abs2 = cv2.convertScaleAbs(sobel2)

fast = cv2.fastNlMeansDenoising(gray, None, 20, 20, 7)

sobel = cv2.Sobel(fast, cv2.CV_64F, 1, 1, ksize=5)

sobel_abs = cv2.convertScaleAbs(sobel)

cv2.imshow('Original', img)
cv2.imshow('Sobel Fast', sobel_abs)
cv2.imshow('Sobel Blur', sobel_abs2)

cv2.waitKey(0)
cv2.destroyAllWindows()
