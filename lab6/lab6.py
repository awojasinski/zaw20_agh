import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cap = cv.VideoCapture('vid1_IR.avi')

thresholdVal = 50
kernel = np.ones((3, 3), np.uint8)

while cap.isOpened():

    ret, frame = cap.read()
    G = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    G = cv.threshold(G, thresholdVal, 255, cv.THRESH_BINARY)[1]

    G = cv.medianBlur(G, 3)
    G = cv.erode(G, kernel, iterations=1)
    G = cv.dilate(G, kernel, iterations=2)
    G = cv.erode(G, kernel, iterations=1)

    retval, labels, stats, centroids = cv.connectedComponentsWithStats(G, connectivity=8)

    if stats.shape[0] > 1:
        tab = stats[1:, 4]
        pi = np.argmax(tab)
        pi = pi + 1
        cv.rectangle(G, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]),
                     (255, 0, 0), 2)

    cv.imshow('IR', G)

    if cv.waitKey(1) & 0xFF == ord('q'):    # przerwanie petli powcisnieciu klawisza ’q'
        break
cap.release()
