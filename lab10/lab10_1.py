import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import scipy.ndimage.filters as filters


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


# ---------------------------------------------
# Fontanna
i1 = cv.imread('pliki_harris/fontanna1.jpg', 0)
i2 = cv.imread('pliki_harris/fontanna2.jpg', 0)

mask_size = 7

h1 = harris(i1, mask_size)
max_i1 = find_max(h1, mask_size, 0.5)

h2 = harris(i2, mask_size)
max_i2 = find_max(h2, mask_size, 0.5)

print_points(max_i1, i1, 'Fontanna 1', 1, 2, 1)
print_points(max_i2, i2, 'Fontanna 2', 1, 2, 2)
plt.show()

# ---------------------------------------------
# Budynek
i1 = cv.imread('pliki_harris/budynek1.jpg', 0)
i2 = cv.imread('pliki_harris/budynek2.jpg', 0)

h1 = harris(i1, mask_size)
max_i1 = find_max(h1, mask_size, 0.5)

h2 = harris(i2, mask_size)
max_i2 = find_max(h2, mask_size, 0.5)

print_points(max_i1, i1, 'Budynek 1', 1, 2, 1)
print_points(max_i2, i2, 'Budynek 2', 1, 2, 2)
plt.show()


