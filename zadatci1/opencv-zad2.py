'''
Pomoću OpenCV-a učitajte sliku koja je snimljena s kamere automobila pod nazivom
“kamera2.jpg”. Ispišite na ekran dimenzije slike. Promijenite veličinu slike na 1000x650.
Prikažite sliku s promijenjenom veličinom, te ispišite na ekran novu veličinu slike
'''

import cv2

img = cv2.imread('kamera2_1.jpg')

print(img.shape)

img_resize = cv2.resize(img, (1000, 650))

cv2.imshow('Resized image', img_resize)
print(img_resize.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()

