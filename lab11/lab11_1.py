import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import scipy.ndimage.filters as filters
import pm


def pts_description(image, pts, size):
    if size % 2 != 0:
        size += 1
    X, Y = image.shape
    pts = list(filter(lambda pt: pt[0] >= size and pt[0] < Y-size and pt[1] >= size and pt[1] < X-size, zip(pts[0], pts[1])))
    l_otoczen = []
    l_wspolrzednych = []
    for i in range(len(pts)):
        opis = (image[pts[i][0]-int(size/2):pts[i][0]+int(size/2), pts[i][1]-int(size/2):pts[i][1]+int(size/2)]).flatten()
        opis_aff = (opis - np.mean(opis)) / np.std(opis)
        l_otoczen.append(opis_aff)
        l_wspolrzednych.append(pts[i])

    wynik_funkcji = list(zip(l_otoczen, l_wspolrzednych))
    return wynik_funkcji


def compare(desc1, desc2, n):
    lst = []
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            tmp = desc2[j][0] - desc1[i][0]
            wynik = np.sum(np.power(tmp, 2))
            lst.append([[i, j], wynik])
    lst.sort(key=lambda x: x[1], reverse=False)
    return lst[0:n]


def harris(image, mask_size):
    k = 0.05

    sobelx = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=mask_size)
    sobely = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=mask_size)
    sobelx2 = sobelx * sobelx
    sobely2 = sobely * sobely
    sobelxy = sobelx * sobely

    I_x2 = cv.GaussianBlur(sobelx2, (mask_size, mask_size), 0)
    I_xy = cv.GaussianBlur(sobelxy, (mask_size, mask_size), 0)
    I_y2 = cv.GaussianBlur(sobely2, (mask_size, mask_size), 0)

    det_M = I_x2 * I_y2 - I_xy * I_xy
    trace_M = I_x2 + I_y2

    H = det_M - k * trace_M * trace_M
    H = cv.normalize(H, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return H


def find_max(image, size, threshold):   # size - rozmiar maski filtra maksymalnego
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)


def print_points(points, image, title, n, m, i):
    plt.subplot(n, m, i)
    plt.imshow(image)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.plot(points[1], points[0], '*', 'm')
    plt.title(title)


img1 = ['pliki_harris/fontanna1.jpg', 'pliki_harris/budynek1.jpg', 'eiffel1.jpg']
img2 = ['pliki_harris/fontanna2.jpg', 'pliki_harris/budynek2.jpg', 'eiffel2.jpg']

for im1, im2 in zip(img1, img2):
    i1 = cv.imread(im1, 0)
    i2 = cv.imread(im2, 0)

    mask_size = 7

    h1 = harris(i1, mask_size)
    max_i1 = find_max(h1, mask_size, 0.5)

    h2 = harris(i2, mask_size)
    max_i2 = find_max(h2, mask_size, 0.5)

    y = max(i1.shape[0], i2.shape[0])
    x = max(i1.shape[1], i2.shape[1])
    plane1 = np.zeros((y, x), dtype=int)
    plane1[0:i1.shape[0], 0:i1.shape[1]] = i1
    plane2 = np.zeros((y, x), dtype=int)
    plane2[0:i2.shape[0], 0:i2.shape[1]] = i2
    i1 = plane1
    i2 = plane2

    desc1 = pts_description(i1, max_i1, 15)
    desc2 = pts_description(i2, max_i2, 15)

    pnt_list = compare(desc1, desc2, 15)

    connections = []
    for i in range(len(pnt_list)):
        xy1 = desc1[pnt_list[i][0][0]][1]
        xy2 = desc2[pnt_list[i][0][1]][1]
        connections.append([[xy1[0], xy1[1]], [xy2[0], xy2[1]]])

    print_points(max_i1, i1, 'Obraz 1', 1, 2, 1)
    print_points(max_i2, i2, 'Obraz 2', 1, 2, 2)
    plt.show()

    pm.plot_matches(i1, i2, connections)
    plt.gray()
    plt.show()
