import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpoints = []  # punkty 3d w przestrzeni (rzeczywsite)
imgpoints = []  # punkty 2d w plaszczyznie obrazu.

for i in range(1, 14):
    img = cv.imread('images_left/left%02d.jpg' % i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    if ret == True:
        #dolaczenie wspolrzednych 3D
        objpoints.append(objp)
        # poprawa lokalizacji punktow (podpikselowo)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # dolaczenie poprawionych punktow
        imgpoints.append(corners2)
        # wizualizacja wykrytych naroznikow
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv.imshow("Corners", img)
    cv.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('ret: ', ret)
print('mtx: ', mtx)
print('dist: ', dist)
print('rvecs: ', rvecs)
print('tvecs: ', tvecs)

h, w = gray.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

cv.imwrite("calibresult.png", dst)
