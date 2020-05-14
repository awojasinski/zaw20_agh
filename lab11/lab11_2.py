import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

SIFT = cv.SIFT_create()

img1 = ['pliki_harris/fontanna1.jpg', 'pliki_harris/fontanna1.jpg', 'pliki_harris/budynek1.jpg', 'eiffel1.jpg']
img2 = ['pliki_harris/fontanna2.jpg', 'pliki_harris/fontanna_pow.jpg', 'pliki_harris/budynek2.jpg', 'eiffel2.jpg']

for im1, im2 in zip(img1, img2):
    i1 = cv.imread(im1, 0)
    i2 = cv.imread(im2, 0)

    i1_pts, i1_descs = SIFT.detectAndCompute(i1, None)
    i2_pts, i2_descs = SIFT.detectAndCompute(i2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(i1_descs, i2_descs, k=2)
    best_matches = [[m] for m, n in matches if m.distance < 0.2 * n.distance]

    plt.figure(1)
    wynik = cv.drawMatchesKnn(i1, i1_pts, i2, i2_pts, best_matches, None, flags=2)
    plt.imshow(cv.cvtColor(wynik, cv.COLOR_BGR2RGB))