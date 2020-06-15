import cv2 as cv
import math
import numpy as np
from scipy.spatial import distance

kernel_size1 = 50 # rozmiar rozkladu
kernel_size2 = 80 # rozmiar rozkladu
mouseX, mouseY = (830, 430)

sigma = kernel_size1 / 6  # odchylenie std
x = np.arange(0, kernel_size1, 1, float)  # wektor poziomy
y = x[:, np.newaxis]  # wektor pionowy
x0 = y0 = kernel_size1 // 2  # wsp. srodka
G1 = 1 / (2 * math.pi * sigma ** 2) * np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

sigma = kernel_size2 / 5  # odchylenie std
x = np.arange(0, kernel_size2, 1, float)  # wektor poziomy
y = x[:, np.newaxis]  # wektor pionowy
x0 = y0 = kernel_size2 // 2  # wsp. srodka
G2 = 1 / (2 * math.pi * sigma ** 2) * np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def track_init(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.rectangle(param, (x-kernel_size1//2, y-kernel_size1//2), (x + kernel_size1//2, y + kernel_size1//2), (0, 255, 0), 2)
        mouseX, mouseY = x, y


def gauss_hist(I_H_roi, G):
    hist_q = np.zeros((256, 1), float)
    for u in range(256):
        mask = I_H_roi == u
        hist_q[u] = np.sum(G[mask])

    hist_q = hist_q / np.max(hist_q)
    return hist_q


def findPos(x, y, I_H, hist):
    while True:
        xS = x - kernel_size2 // 2
        yS = y - kernel_size2 // 2

        I_H_roi = I_H[yS:yS + kernel_size2, xS:xS + kernel_size2]

        hist_new = gauss_hist(I_H_roi, G2)
        hist_mul = np.sqrt(hist * hist_new)

        mask = np.zeros((kernel_size2, kernel_size2), float)
        for ii in range(0, kernel_size2):
            for jj in range(0, kernel_size2):
                mask[ii, jj] = hist_mul[I_H_roi[ii, jj]]
        mask = mask*G2

        M = cv.moments(mask)
        imgX = int(M["m10"] / M["m00"])
        imgY = int(M["m01"] / M["m00"])

        newX = xS + imgX
        newY = yS + imgY
        dist = distance.euclidean((newX, newY), (x,y))
        if dist < 2:
            break
        x = newX
        y = newY

    return newX, newY, hist_new


for i in range(100, 200):
    img_name = 'track_seq/track00' + str(i) + '.png'
    I = cv.imread(img_name)
    I_HSV = cv.cvtColor(I, cv.COLOR_BGR2HSV)
    I_H = I_HSV[:, :, 0]
    if i == 100:
        cv.namedWindow('Tracking')
        cv.setMouseCallback('Tracking', track_init, param=I)

        # Pobranie klawisza
        while (1):
            cv.imshow('Tracking', I)
            k = cv.waitKey(20) & 0xFF
            if k == 27:
                cv.destroyAllWindows()
                break
        x = mouseX
        y = mouseY
        xS = x - kernel_size1 // 2
        yS = y - kernel_size1 // 2
        I_H_roi = I_H[yS:yS+kernel_size1, xS:xS+kernel_size1]
        hist = gauss_hist(I_H_roi, G1)
    else:
        x, y, hist = findPos(x, y, I_H, hist)
    cv.rectangle(I, (x-kernel_size1//2, y-kernel_size1//2), (x+kernel_size1//2, y+kernel_size1//2), (0, 255, 0), 2)
    cv.imshow('Ttacking', I)
    cv.waitKey(50)
