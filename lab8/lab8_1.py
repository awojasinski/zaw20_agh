from scipy.spatial import distance

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
    theta = np.rad2deg(np.arctan2(sobely, sobelx))

    return im, sobel, theta


im, sobel, theta = imageProcessing('trybik.jpg')

# Binaryzacja i kontury
thresh = cv.threshold(im, 100, 255, 0)[1]
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# Rysowanie konturów
zeros = np.zeros(shape=im.shape, dtype=float)
cv.drawContours(zeros, contours, -1, (255, 0, 0))
plt.figure()
plt.gray()
plt.imshow(zeros)
plt.title('Kontur')
plt.axis('off')

# Rysowanie Orientacji gradientu i gradientu
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 2)
plt.gray()
plt.imshow(theta)
plt.title('Orientacja gradientu')
plt.axis('off')
plt.subplot(1, 2, 1)
plt.gray()
plt.imshow(sobel)
plt.title('Gradient')
plt.axis('off')

# Wyznaczenie środka ciężkości
center = [0, 0]
m = cv.moments(thresh, 1)
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
        xl = (center[0] - x)
        yl = (center[1] - y)
        # Długość wektora
        dist = distance.euclidean(center, [x, y])
        # Kąt pomiędzy wektorem a OX
        a = int(np.rad2deg(np.arctan2(yl, xl)))
        Rtable[n].append((dist, a))

im2, sobel2, theta2 = imageProcessing('trybiki2.jpg')

# Rysowanie Orientacji gradientu i gradientu
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 2)
plt.gray()
plt.imshow(theta2)
plt.title('Orientacja gradientu')
plt.axis('off')
plt.subplot(1, 2, 1)
plt.gray()
plt.imshow(sobel2)
plt.title('Gradient')
plt.axis('off')

accu = np.zeros(im2.shape)  # Inicjalizacja tablicy akumulatorów

for x in range(sobel2.shape[0]):
    for y in range(sobel2.shape[1]):
        if sobel2[x, y] > 0.5:
            for p in Rtable[int(theta2[x, y])]:
                r = p[0]
                fi = p[1]
                x_c = int(x + r * np.cos(np.deg2rad(fi)))
                y_c = int(y + r * np.sin(np.deg2rad(fi)))
                if x_c < accu.shape[0] and y_c < accu.shape[1]:
                    accu[x_c, y_c] += 1

max_hough = np.where(accu.max() == accu)
print(max_hough)

plt.figure()
plt.gray()
plt.imshow(accu * 255 / accu.max())
plt.title('Przestrzeń Hougha')
plt.axis('off')

plt.figure()
plt.imshow(~im2)
plt.plot(max_hough[1], max_hough[0], '*m')
plt.title('Efekt końcowy')
plt.axis('off')

plt.show()
