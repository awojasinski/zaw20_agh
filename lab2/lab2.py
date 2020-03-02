import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


kernel = np.ones((2, 2), np.int)

for i in range(300, 1099, 1):
    I_prev = cv.imread('pedestrants/input/in%06d.jpg' % i)
    I = cv.imread('pedestrants/input/in%06d.jpg' % int(i+1))
    '''
    cv.imshow('I', I)
    cv.waitKey(10)
    '''
    I_G = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
    I_prev_G = cv.cvtColor(I_prev, cv.COLOR_BGR2GRAY)
    I_G = I_G.astype(int)
    I_prev_G = I_prev_G.astype(int)
    I_mov = cv.absdiff(I_G, I_prev_G)
    I_mov_B = 1 * (I_mov > 20)
    I_mov_B = I_mov_B * 255
    I_mov_B = I_mov_B.astype(np.uint8)
    I_mov_B_median = cv.medianBlur(I_mov_B, 5)
    I_mov_B_erosion = cv.erode(I_mov_B_median, kernel)
    I_mov_B_dilatation = cv.dilate(I_mov_B_erosion, kernel)

    retval, labels, stats, centroids = cv.connectedComponentsWithStats(I_mov_B_dilatation)
    cv.imshow("Labels", np.uint8(labels / retval * 255))
    cv.imshow('I', I_mov_B_dilatation)

    cv.waitKey(10)