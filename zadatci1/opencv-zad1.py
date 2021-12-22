'''
Pomoću OpenCV-a učitajte sliku koja je snimljena s kamere automobila pod nazivom
“kamera2.jpg”. Sliku učitajte u nijansama sive boje. Ispišite na ekran dimenzije slike (visinu i
širinu). Prikažite sliku, te spremite ju pod nazivom “kamera2_crno_bijela.jpg”.
'''

import cv2

gray_img = cv2.imread('kamera2_1.jpg', 0)

h, w = gray_img.shape

print('Image height ', h)
print('Image width ', w)

cv2.imshow('Gray image', gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('kamera2_crno_bijela.jpg', gray_img)