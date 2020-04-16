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


map_I = cv.imread("Aegeansea.jpg")
map = cv.cvtColor(map_I, cv.COLOR_BGR2HSV)

map_h = (~map[:, :, 0] > 190) * 1
map_s = (map[:, :, 1] > 30) * 1
map = map_h * map_s
map = map.astype(np.uint8)

contours, hierarchy = cv.findContours(map, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours = list(filter(lambda el: 15 < el.shape[0] < 3000, contours))

islands_list = os.listdir('imgs')

for island in islands_list:
    island_img_path = 'imgs/' + island
    I = cv.imread(island_img_path)
    I_G = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
    I_G = cv.bitwise_not(I_G)

    thresh = cv.threshold(I_G, 127, 255, 0)[1]
    c, hier = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    xy, center = contour_norm(c[0])

    img = np.zeros(shape=(map.shape[0], map.shape[1], 3), dtype=np.uint8)
    img.fill(255)
    dist_island = []
    for n, cnt in enumerate(contours):
        cv.drawContours(img, cnt, -1, (0, 0, 0))
        xy_c, center_c = contour_norm(cnt)
        cv.putText(map_I, str(n), (int(center_c[0]), int(center_c[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128))
        dist_island.append(hausdorf(xy, xy_c))

    min_haus = np.argmin(dist_island)
    print("Island ", min_haus, ". dist = ", dist_island[min_haus])

    island_name = island_img_path.split('.')[0].split('_')[1]
    _, island_coord = contour_norm(contours[min_haus])

    cv.putText(map_I, island_name, (int(island_coord[0]), int(island_coord[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    contours.pop(min_haus)

plt.imshow(map, cmap=plt.cm.gray)
plt.axis('off')
plt.show(block=True)
map = map * 255
cv.imwrite('binary_map.png', map)

plt.imshow(img)
plt.axis('off')
plt.show(block=True)
cv.imwrite('islands_cnts.png', img)

cv.imshow("map", map_I)
cv.imwrite('names_map.png', map_I)
cv.waitKey(0)
