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

    # Orientacja sobel2ientu
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

# Rysowanie Orientacji sobel2ientu i sobel2ientu
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
        # Orientacja sobel2ientu w punkcie konturu
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

# Rysowanie Orientacji sobel2ientu i sobel2ientu
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
new_accu = np.zeros(accu.shape + (36, ))

for x in range(sobel2.shape[0]):
    for y in range(sobel2.shape[1]):
        for df in range(0, 360, 10):
            if sobel2[x, y] > 0.5:
                idx_rtable = int(theta2[x, y]) + df
                if idx_rtable >= 360:
                    idx_rtable = int(theta2[x, y]) + df - 360
                for one in Rtable[idx_rtable]:
                    r = one[0]
                    fi = one[1]
                    x_c = int(x + r * np.cos(np.deg2rad(fi) + df))
                    y_c = int(y + r * np.sin(np.deg2rad(fi) + df))
                    if x_c < new_accu.shape[0] and y_c < new_accu.shape[1]:
                        new_accu[x_c][y_c][int(df/10)] += 1

delta = 30
maximum = list()

for n in range(0, 4):
    temp_max = np.unravel_index(np.argmax(new_accu), new_accu.shape)
    maximum.append(temp_max)
    print(maximum[n])
    new_accu[temp_max[0]-delta:temp_max[0]+delta, temp_max[1]-delta:temp_max[1]+delta, :] = 0

print(maximum)

plt.figure()
plt.imshow(~im2)
for i in maximum:
    plt.plot(i[1], i[0], '*m')
plt.title('Efekt końcowy')
plt.axis('off')

plt.show()
