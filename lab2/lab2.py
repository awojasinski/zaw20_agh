import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


kernel = np.ones((3, 3), np.int)

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
    I_mov_B = 1 * (I_mov > 15)
    I_mov_B = I_mov_B * 255
    I_mov_B = I_mov_B.astype(np.uint8)
    I_mov_B_median = cv.medianBlur(I_mov_B, 7)
    I_mov_B_erosion = cv.erode(I_mov_B_median, kernel, iterations=1)
    I_mov_B_dilatation = cv.dilate(I_mov_B_erosion, kernel, iterations=1)

    retval, labels, stats, centroids = cv.connectedComponentsWithStats(I_mov_B_dilatation)

    cv.imshow("Labels", np.uint8(labels / retval * 255))
    cv.imshow('I', I_mov_B_dilatation)

    I_VIS = I
    if stats.shape[0] > 1:
        tab = stats[1:, 4]
        pi = np.argmax(tab)
        pi = pi + 1

        cv.rectangle(I_VIS, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0]+stats[pi, 2], stats[pi, 1]+stats[pi, 3]), (255, 0, 0), 2)# wypisanie informacji o polu i numerze najwiekszego elementu
        cv.putText(I_VIS, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv.putText(I_VIS, "%d" %pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    cv.imshow("frames", I_VIS)

    cv.waitKey(10)