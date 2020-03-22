import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate_Camera(imagesPath):
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    objpoints = []  # punkty 3d w przestrzeni (rzeczywsite)
    imgpoints = []  # punkty 2d w plaszczyznie obrazu.

    path = imagesPath + '/' + imagesPath.split('_')[1] + '%02d.jpg'
    for i in range(1, 14):
        img = cv.imread(path % i)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist, imgpoints, objpoints, gray,


mtx_L, dist_L, imgpoints_L, objpoints, gray_l = calibrate_Camera('images_left')
mtx_R, dist_R, imgpoints_R, _, gray_r = calibrate_Camera('images_right')

if len(imgpoints_L) != len(imgpoints_R):
    difference = len(imgpoints_L) - len(imgpoints_R)
    if difference > 0:
        for i in range(difference):
            imgpoints_L.pop(i)
            objpoints.pop(i)
    else:
        difference = abs(difference)
        for i in range(difference):
            imgpoints_R.pop(i)



retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_L,
                                                                                                imgpoints_R, mtx_L,
                                                                                                dist_L, mtx_R, dist_R,
                                                                                                gray_r.shape[::-1])
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(cameraMatrix1, distCoeffs1,
                                                                 cameraMatrix2, distCoeffs2,
                                                                 gray_r.shape[::-1], R, T)
map1_L, map2_L = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray_l.shape[::-1], cv.CV_16SC2)
map1_R, map2_R = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray_r.shape[::-1], cv.CV_16SC2)

for i in range(1, 14):

    img_l = cv.imread('images_left/left%02d.jpg' % i)
    img_r = cv.imread('images_right/right%02d.jpg' % i)

    cv.imshow("L", img_l)
    cv.imshow("R", img_r)

    dst_L = cv.remap(img_l, map1_L, map2_L, cv.INTER_LINEAR)
    dst_R = cv.remap(img_r, map1_R, map2_R, cv.INTER_LINEAR)

    N, XX, YY = dst_L.shape[::-1] # pobranie rozmiarow obrazka (kolorowego)
    visRectify = np.zeros((YY, XX*2, N), np.uint8) # utworzenie nowego obrazka o szerokosci x2
    visRectify[:, 0:640:, :] = dst_L # przypisanie obrazka lewego
    visRectify[:, 640:1280:, :] = dst_R # przypisanie obrazka prawego
    # Wyrysowanie poziomych linii
    for y in range(0, 480, 10):
        cv.line(visRectify, (0, y), (1280, y), (255, 0, 0))
        cv.imshow('visRectify', visRectify) #wizualizacja
        cv.waitKey(10)
