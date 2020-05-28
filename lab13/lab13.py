import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass as center_of_mass

kernel_size = 45
mouseX, mouseY = (830, 430)


def track_init(event, x, y, flag, param):
    global mouseX, mouseY
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.rectangle(param, (x-kernel_size//2, y-kernel_size//2), (x + kernel_size//2, y + kernel_size//2), (0, 255, 0), 2)
        mouseX, mouseY = x, y

def gen_gauss(kernel_size, mouseX, mouseY, I):
    # Generowanie Gaussa
    sigma = kernel_size / 6  # odchylenie std
    x = np.arange(0, kernel_size, 1, float)  # wektor poziomy
    y = x[:, np.newaxis]  # wektor pionowy
    x0 = y0 = kernel_size // 2  # wsp. srodka
    G = 1 / (2 * math.pi * sigma ** 2) * np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    xS = mouseX - kernel_size // 2
    yS = mouseY - kernel_size // 2

    I_HSV = cv.cvtColor(I, cv.COLOR_BGR2HSV)
    I_H = I_HSV[:, :, 0]
    hist_q = np.zeros((256, 1), float)
    for jj in range(0, kernel_size):
        for ii in range(0, kernel_size):
            pixel_H = I_H[yS + jj, xS + ii]
            hist_q[pixel_H] += G[jj, ii]

    hist_q = hist_q / np.amax(hist_q)
    I_H_part = I_H[yS : yS + kernel_size, xS : xS + kernel_size]
    return hist_q, I_H_part


# Wczytanie pierwszego obrazka
I = cv.imread('track_seq/track00100.png')
cv.namedWindow('Tracking')
cv.setMouseCallback('Tracking', track_init, param=I)


# Pobranie klawisza
while(1):
    cv.imshow('Tracking', I)
    k = cv.waitKey(20) & 0xFF
    if k == 27:
        break

hist_q_1, I_H_1 = gen_gauss(kernel_size=kernel_size, mouseX=mouseX, mouseY=mouseY, I=I)
yS = mouseY
xS = mouseX

for i in range(101, 200):
    img_name = 'track_seq/track00' + str(i) + '.png'
    I_next = cv.imread(img_name)
    cv.rectangle(I_next, (xS - kernel_size // 2, yS - kernel_size // 2), (xS + kernel_size // 2, yS + kernel_size // 2),
                  (0, 255, 0), 2)

    hist_q_2, I_H_2 = gen_gauss(kernel_size=kernel_size, mouseX=xS, mouseY=yS, I=I_next)

    x_na_potem = xS - kernel_size // 2
    y_na_potem = yS - kernel_size // 2

    bhatta = np.sqrt(hist_q_2 * hist_q_1)
    I = I_next

    Wynik = np.zeros((kernel_size, kernel_size), dtype=float)
    for jj in range(0, kernel_size):
        for ii in range(0, kernel_size):
            pixel_H = I_H_2[jj, ii]
            Wynik[jj, ii] = bhatta[pixel_H] * hist_q_2[pixel_H]

    xS, yS = center_of_mass(Wynik)
    xS = int(xS + x_na_potem)
    yS = int(yS + y_na_potem)

    print(xS, yS)
    cv.imshow('Tracking', I_next)
    cv.waitKey(10)
