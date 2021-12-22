import cv2
import numpy as np

video = cv2.VideoCapture('dashcam_video.mp4')

x, y, w, h = 390, 295, 530 - 390, 420 - 295
track_window = (x, y, w, h)

_, frame = video.read()

roi = frame[y: y+h, x: x+w, :]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

green_lower = np.array([100, 20, 20])
green_upper = np.array([130, 180, 130])

mask = cv2.inRange(roi_hsv, green_lower, green_upper)
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

    if cv2.waitKey(3) == 27:
        break

video.release()
cv2.destroyAllWindows()
