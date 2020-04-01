import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cap = cv.VideoCapture('vid1_IR.avi')

thresholdVal = 45
error_val = 30
kernel = np.ones((2, 2), np.uint8)

while cap.isOpened():

    ret, frame = cap.read()
    G = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    B = cv.threshold(G, thresholdVal, 255, cv.THRESH_BINARY)[1]
    B = cv.medianBlur(B, 3)
    B = cv.erode(B, kernel, iterations=1)
    B = cv.dilate(B, kernel, iterations=3)

    # Indeksacja obiektów
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(B)
    labels_img = np.uint(labels / retval*255)
    labels_img = labels_img.astype(np.float32)
    stats = stats[np.argsort(stats[:, 4])]  # sortowanie według pola powierzchni

    # prostokąty + pole sprawdzające czy prostokąt był analizowany
    rects = np.c_[stats, np.zeros((stats.shape[0], ))]

    I = frame
    if stats.shape[0] > 1:
        # pętla po wszystkich prostokątach poza ramką wokół obrazu
        for rec in range(stats.shape[0]-1, 0, -1):
            # sprawdzenie czy prostokąt nie jest dużo dłuższy niż wyższy  (np. świetlówka)
            if (2*stats[rec, 3] > stats[rec, 2]) and rects[rec, 5] != 1:
                if stats[rec, 4] > 900 and stats[rec, 4] < 100000:
                    left = stats[rec, 0]
                    right = stats[rec, 0] + stats[rec, 2]
                    top = stats[rec, 1]
                    bottom = stats[rec, 1] + stats[rec, 3]
                    rects[rec, 5] = 1

                    neighbors = []  # inicjalizacja tablicy przechowującej prostokąty należące do jednej postaci
                    neighbors.append((left, top))
                    neighbors.append((right, bottom))
                    for rec_n in range(stats.shape[0]-1, 0, -1):
                        if rects[rec_n, 5] != 1 and 10 < stats[rec_n, 4] < 100000:
                            m = stats[rec_n, 0]
                            n = stats[rec_n, 1]
                            r_witdh = stats[rec_n, 2]
                            r_height = stats[rec_n, 3]
                            if r_witdh < 2*r_height:
                                # sprawdzenie czy nowy prostokąt możee należeć do tego samego obrysu
                                # 1. jego punkt początkowy musi mieścić się w zakresie poprzedniego+/-error
                                # 2. na wysokość nie powinien być większy od poprzedniego i nie powinien wykraczać poza jego granice+/-error
                                # szukamy prostokątów które należą do większego (poszukujemy sąsiednich małych które przynależą do większego)
                                if (left-error_val <= m <= right+error_val) and ((top-error_val <= n <= bottom+error_val) or (top-error_val <= n+r_height <=bottom+error_val)):
                                    neighbors.append((m, n))
                                    neighbors.append((m+r_witdh, n+r_height))
                                    rects[rec_n, 5] = 1 # prostokąt został przeanalizowany i przydzielony do obrysu
                    x, y, w, h = cv.boundingRect(np.asarray(neighbors)) # tworzymy prostokąt wokół prostokątów dodanych do tablicy

                    cv.rectangle(I, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv.imshow('Termowizja', I)
    cv.imshow('Labels', labels_img)

    if cv.waitKey(10) & 0xFF == ord('q'):    # przerwanie petli powcisnieciu klawisza ’q'
        break
cap.release()
