import numpy as np
import cv2 as cv
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt


def contour_norm(c):
    center = [0, 0]
    xy = c[:, 0, :]
    xy = xy.astype(float)

    m = cv.moments(c, 1)
    if m["m00"] != 0:
        center[0] = int(m["m10"] / m["m00"])  # Obliczenie współrzędnej X
        center[1] = int(m["m01"] / m["m00"])  # Obliczenie współrzędnej Y

    xy[:, 0] = xy[:, 0] - center[0]
    xy[:, 1] = xy[:, 1] - center[1]

    max_dist = 0
    for i in xy:
        for j in xy:
            dist = distance.euclidean(i, j)
            if dist > max_dist:
                max_dist = dist

    xy[:, 0] = xy[:, 0] / max_dist
    xy[:, 1] = xy[:, 1] / max_dist
    return xy, center


def dH(xy_c1, xy_c2):
    dist = []
    for i in xy_c1:
        dist_point = []
        for j in xy_c2:
            dist_point.append(distance.euclidean(i, j))
        max_dist_point = min(dist_point)
        dist.append(max_dist_point)
    max_dist = max(dist)
    return max_dist


def hausdorf(xy_c1, xy_c2):
    return max(dH(xy_c1, xy_c2), dH(xy_c2, xy_c1))


I = cv.imread("ithaca_q.bmp")
I_G = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
I_G = cv.bitwise_not(I_G)

thresh = cv.threshold(I_G, 127, 255, 0)[1]
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
xy, center = contour_norm(contours[0])

imgs = os.listdir("imgs")
dist_imgs = []
for n, i in enumerate(imgs):
    img_path = 'imgs/' + i
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.bitwise_not(img)
    img = cv.threshold(img, 127, 255, 0)[1]

    c, hier = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    xy_c, center_c = contour_norm(c[0])
    dist_imgs.append(hausdorf(xy, xy_c))
    print(i, '. ', dist_imgs[n])

min_haus = np.argmin(dist_imgs)
print(imgs[min_haus], ', dist= ', dist_imgs[min_haus])

cv.drawContours(I, contours, (255, 0, 0))
cv.imshow("I", I)

cv.waitKey(0)
