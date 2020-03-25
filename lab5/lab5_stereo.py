import cv2 as cv
import numpy as np

# Kryteria dotyczące przerwania obliczen subpikseli szachwnicy
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# Przygotowanie punktów 2D
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Tablice do przechowywania punktow obiektów (3D) i na obrazie (2D) dla wszystkich obrazow
objpoints = []
imgpoints_L = []
imgpoints_R = []

for i in range(1, 13):
    # Wczytaj obraz i konwersja do obrazu monochromatycznego
    img_L = cv.imread('images_left/left%02d.jpg' % i)
    img_R = cv.imread('images_right/right%02d.jpg' % i)
    gray_L = cv.cvtColor(img_L, cv.COLOR_BGR2GRAY)
    gray_R = cv.cvtColor(img_R, cv.COLOR_BGR2GRAY)

    # Wyszukiwanie wzorca kalibracyjnego na zdjęciu
    ret_L, corners_L = cv.findChessboardCorners(gray_L, (7,6), None)
    ret_R, corners_R = cv.findChessboardCorners(gray_R, (7,6), None)

    # Jesli znaleziono na obrazie punkty
    if ret_L == True and ret_R == True:
        # Dolaczenie wspolrzednych 3D
        objpoints.append(objp)

        # Poprawa lokalizacji punktow przecięcia szachownicy
        corners2 = cv.cornerSubPix(gray_L, corners_L, (11, 11), (-1, -1), criteria)
        imgpoints_L.append(corners2)
        cv.drawChessboardCorners(img_L, (7, 6), corners2, ret_L)
        corners2 = cv.cornerSubPix(gray_R, corners_R, (11, 11), (-1, -1), criteria)
        imgpoints_R.append(corners2)
        cv.drawChessboardCorners(img_R, (7, 6), corners2, ret_R)

    # Rozdzielczość obrazu
    h, w = img_L.shape[:2]

ret_L, mtx_L, dist_L, rvecs_L, tvecs_L = cv.calibrateCamera(objpoints, imgpoints_L, gray_L.shape[::-1], None, None)
ret_R, mtx_R, dist_R, rvecs_R, tvecs_R = cv.calibrateCamera(objpoints, imgpoints_R, gray_R.shape[::-1], None, None)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(
    objpoints, imgpoints_L, imgpoints_R, mtx_L, dist_L, mtx_R, dist_R, gray_L.shape[::-1])

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(cameraMatrix1, distCoeffs1,
                                                                 cameraMatrix2, distCoeffs2,
                                                                 gray_R.shape[::-1], R, T)

map1_L, map2_L = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray_L.shape[::-1], cv.CV_16SC2)
map1_R, map2_R = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray_L.shape[::-1], cv.CV_16SC2)

for i in range(1, 13):
    img_L = cv.imread('images_left/left%02d.jpg' % i)
    img_R = cv.imread('images_right/right%02d.jpg' % i)


    dst_L = cv.remap(img_L, map1_L, map2_L, cv.INTER_LINEAR)
    dst_R = cv.remap(img_R, map1_R, map2_R, cv.INTER_LINEAR)

    N, XX, YY = dst_L.shape[::-1] # pobranie rozmiarow obrazu
    visRectify = np.zeros((YY, XX*2, N), np.uint8) # utworzenie nowego obrazu
    visRectify[:, 0:640:, :] = dst_L # przypisanie obrazu lewego
    visRectify[:, 640:1280:, :] = dst_R # przypisanie obrazu prawego
    # Wyrysowanie poziomych linii
    for y in range(0, 480, 10):
        cv.line(visRectify, (0, y), (1280, y), (255, 0, 0))


    cv.imshow('visRectify', visRectify) # Wyświetlenie obrazu
    cv.waitKey(300)

cv.imwrite('stereo.jpg', visRectify)
