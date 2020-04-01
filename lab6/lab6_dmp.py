import cv2 as cv
import numpy as np

thresholdVal = 40
r_height = 194
r_width = 64
dim = (r_width, r_height)

DMP = np.empty(shape=(r_height, r_width))

for img in range(1, 50):
    I = cv.imread('pedestrian/sample_%06d.png' % img)
    I = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
    resized = cv.resize(I, dim)
    bin = resized > thresholdVal
    thresh = cv.threshold(resized, thresholdVal, 1, cv.THRESH_BINARY)[1]
    cv.imshow("tr", thresh)
    DMP = DMP + bin

cv.imwrite("dmp.png", DMP)