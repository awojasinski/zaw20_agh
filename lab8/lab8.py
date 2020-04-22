import cv2 as cv
import numpy as np
from math import pi
import matplotlib.pyplot as plt

kernel = np.ones((5, 5), np.uint8)

# Wczytanie obrazu
im = cv.imread('trybik.jpg')
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im = cv.bitwise_not(im)

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

sobelx = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=5)     # Filtr Sobela X
sobely = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=5)     # Filtr Sobela Y
sobel = np.sqrt(sobelx**2 + sobely**2)              # Obliczenie pierwiastka sumy kwadratów
sobel = sobel/np.amax(sobel)                        # Normalizacja

theta = np.arctan2(sobely, sobelx)
theta = theta*(theta >= 0) + (theta+2*pi)*(theta < 0)
theta = (theta*180)/pi

Rtable = [[] for i in range(360)]

for cnt in contours:
    for p in cnt:
        # Współrzędne punktu konturu
        x = p[0][0]
        y = p[0][1]
        # Orientacja gradientu w punkcie konturu
        n = int(theta[x, y])
        # Długości odcinków wzdlędem punktu referencyjnego
        xl = (x-center[0])
        yl = (center[1]-y)
        # Długość wektora
        l = np.sqrt(xl**2+yl**2)
        # Kąt pomiędzy wektorem a OX
        a = np.arctan2(yl, xl)
        a = a*(a >= 0) + (a+2*pi)*(a < 0)
        a = (a*180)/pi
        vect = [l, a]
        Rtable[n].append(vect)


cv.imshow('I', im)
cv.imshow('B', thresh)
cv.imshow('Sobel', sobel)
cv.waitKey(0)
