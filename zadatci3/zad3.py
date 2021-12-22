import numpy as np
import cv2

def detect_car(gray_image, gray_frame):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray_image, None)
    kp2, des2 = sift.detectAndCompute(gray_frame, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = list()
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])


    if len(good_matches) > 5:
        src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = gray_image.shape

        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        bb_points = np.int32(dst).reshape(-1, 2)

        x, y  = bb_points[0]
        w = abs(x - bb_points[bb_points[:, 0].argmax()][0])
        h = abs(y - bb_points[bb_points[:, 1].argmax()][1])

        return x, y, w, h

    else:
        return 0, 0, 0, 0


def track_car(frame, x, y, w, h):
    green_lower = np.array([100, 20, 20])
    green_upper = np.array([130, 180, 130])

    roi = frame[y: y + h, x: x + w,:]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(roi_hsv, green_lower, green_upper)

    roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])
    roi_hist = cv2.normalize(roi_hist, None, 0, 255, cv2.NORM_MINMAX)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv_frame], [0], roi_hist, [0, 180], 1)

    track_window = (x, y, w, h)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    _, track_window = cv2.meanShift(dst, track_window, criteria)

    x, y, w, h = track_window

    return x, y, w, h


cap = cv2.VideoCapture('dashcam_video.mp4')
gray_img = cv2.imread('car.png', 0)

x, y, w, h = 0, 0, 0, 0

k = 0

while True:
    ret, frame = cap.read()

    if ret == False:
        break

    if k == 0 or k % 10 == 0:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, w, h = detect_car(gray_img, gray_frame)
    else:
        if w != 0 and h != 0:
            x, y, w, h = track_car(frame, x, y, w, h)

    k += 1

    tracking_img = frame
    if w != 0 and h != 0:
        tracking_img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Tracked car", tracking_img)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
