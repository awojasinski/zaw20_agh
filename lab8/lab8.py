from math import pi

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def imageProcessing(filename):
    # Wczytanie obrazu
    im = cv.imread(filename)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.bitwise_not(im)

    sobelx = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=5)  # Filtr Sobela X
    sobely = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=5)  # Filtr Sobela Y
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)  # Obliczenie pierwiastka sumy kwadratów
    sobel = sobel / np.amax(sobel)  # Normalizacja

    # Orientacja gradientu
    theta = np.arctan2(sobely, sobelx)
    theta = theta * (theta >= 0) + (theta + 2 * pi) * (theta < 0)
    theta = (theta * 180) / pi

    return im, sobel, theta


kernel = np.ones((5, 5), np.uint8)

im, sobel, theta = imageProcessing('trybik.jpg')

# Binaryzacja i kontury
thresh = cv.threshold(im, 55, 255, 0)[1]
thresh = cv.dilate(thresh, kernel)
thresh = cv.erode(thresh, kernel)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# Wyznaczenie środka ciężkości
center = [0, 0]
m = cv.moments(contours[0], 1)
if m["m00"] != 0:
    center[0] = int(m["m10"] / m["m00"])  # Obliczenie współrzędnej X
    center[1] = int(m["m01"] / m["m00"])  # Obliczenie współrzędnej Y

Rtable = [[] for i in range(360)]

for cnt in contours:
    for p in cnt:
        # Współrzędne punktu konturu
        x = p[0][0]
        y = p[0][1]
        # Orientacja gradientu w punkcie konturu
        n = int(theta[x, y])
        # Długości odcinków wzdlędem punktu referencyjnego
        xl = (x - center[0])
        yl = (center[1] - y)
        # Długość wektora
        l = np.sqrt(xl ** 2 + yl ** 2)
        # Kąt pomiędzy wektorem a OX
        a = np.arctan2(yl, xl)
        a = a * (a >= 0) + (a + 2 * pi) * (a < 0)
        a = (a * 180) / pi
        vect = [l, a]
        Rtable[n].append(vect)

im2, sobel2, theta2 = imageProcessing('trybiki2.jpg')

accu = np.zeros(im2.shape)  # Inicjalizacja tablicy akumulatorów

for x in range(sobel2.shape[0]):
    for y in range(sobel2.shape[1]):
        if sobel2[x, y] > 0.5:
            for one in Rtable[int(theta2[x, y])]:
                r = one[0]
                fi = one[1]
                x_c = int(x + r * np.cos(fi))
                y_c = int(y + r * np.sin(fi))
                if x_c < accu.shape[0] and y_c < accu.shape[1]:
                    accu[x_c, y_c] += 1

max_hough = np.where(accu.max() == accu)
print(max_hough)

plt.figure()
plt.gray()
plt.imshow(accu * 255 / accu.max())
plt.title('Przestrzeń Hougha')

plt.figure()
plt.imshow(~im2)
plt.plot(max_hough[1], max_hough[0], '*m')
plt.title('Efekt końcowy')

plt.show()

cv.imshow('I', im)
cv.imshow('B', thresh)
cv.imshow('Sobel', sobel)
cv.waitKey(0)
