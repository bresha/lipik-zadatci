import cv2
import numpy as np

MIN_MATCH_COUNT = 5

cap = cv2.VideoCapture('dashcam_video.mp4')

_, frame = cap.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

gray_img = cv2.imread('car.png', 0)

sift = cv2.SIFT_create()

kp1, desc1 = sift.detectAndCompute(gray_img, None)
kp2, desc2 = sift.detectAndCompute(gray_frame, None)


matcher = cv2.BFMatcher()
matches = matcher.knnMatch(desc1, desc2, 2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append([m])

good_matches2 = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches2.append(m)

matched_img = cv2.drawMatchesKnn(gray_img, kp1, gray_frame, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matched', matched_img)

if len(good_matches2) > MIN_MATCH_COUNT:
    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches2]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches2]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    h, w = gray_img.shape
    pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1] ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    bb_points = np.int32(dst.reshape(-1, 2))
    x, y = bb_points[0]
    w = abs(x - bb_points[bb_points[:, 0].argmax()][0])
    h = abs(y - bb_points[bb_points[:, 1].argmax()][1])

    #img_detection = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2)
    img_detection = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('Detection', img_detection)

cv2.waitKey()
cv2.destroyAllWindows()