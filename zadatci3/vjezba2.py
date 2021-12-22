import cv2
import numpy as np

video = cv2.VideoCapture('car_meanshift.mp4')

x, y, w, h = 223, 202, 34, 20
track_window = (x, y, w, h)

_, frame = video.read()

roi = frame[y: y+h, x: x+w, :]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

red_lower = np.array([170, 50, 50])
red_upper = np.array([180, 255, 255])

mask = cv2.inRange(roi_hsv, red_lower, red_upper)
roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])

roi_hist = cv2.normalize(roi_hist, None, 0, 255, cv2.NORM_MINMAX)

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:

    ret, frame = video.read()

    if frame is None:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv_frame], [0], roi_hist, [0, 180], 1)

    _, track_window = cv2.meanShift(dst, track_window, criteria)

    x, y, w, h = track_window

    tracking_iamge = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('Tracked', tracking_iamge)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
