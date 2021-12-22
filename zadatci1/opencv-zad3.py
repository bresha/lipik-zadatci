'''
Pomoću OpenCV-a učitajte sliku koja je snimljena s kamere automobila pod nazivom
“kamera2.jpg”. Promijenite vrijednost piksela koji čine dio registracijske oznake automobila koji
se nalazi ispred nas u žutu boju.
'''

import cv2

img = cv2.imread('kamera2_1.jpg')

img[273: 282, 143 : 154] = [0, 255, 255]

cv2.imshow('Original', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
