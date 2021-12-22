import cv2
import numpy as np

video = cv2.VideoCapture('car_meanshift.mp4')

backSub = cv2.createBackgroundSubtractorMOG2()

if not video.isOpened():
    print('Cannot open video capture')
    exit(1)

while True:

    ret, frame = video.read()

    if frame is None:
        break
    
    binary = backSub.apply(frame)

    kernel = np.ones((5,5), dtype=np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Video', frame)
    cv2.imshow('Morph', morph)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
