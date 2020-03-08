import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

for img_source in ['highway', 'office', 'pedestrants']:

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    kernel = np.ones((2, 2), np.int)
    threshold_value = 10

    f = open(img_source + '/temporalROI.txt','r') # otwarcie pliku
    line = f.readline() # odczyt lini
    roi_start, roi_end = line.split() # rozbicie lini na poszczegolne

    roi_start = int(roi_start)
    roi_end = int(roi_end)
    step_frame = 1

    I_prev = cv.imread(img_source + '/input/in%06d.jpg' % roi_start)

    for i in range(roi_start, roi_end, step_frame):
        
        I = cv.imread(img_source + '/input/in%06d.jpg' % i)
        I_GT = cv.imread(img_source + '/groundtruth/gt%06d.png' % i)

        I_G = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
        I_prev_G = cv.cvtColor(I_prev, cv.COLOR_BGR2GRAY)
        I_GT = cv.cvtColor(I_GT, cv.COLOR_BGR2GRAY)
        I_G = I_G.astype(int)
        I_prev_G = I_prev_G.astype(int)
        I_mov = cv.absdiff(I_G, I_prev_G)
        
        # binaryzacja obrazu
        I_mov_B = 1 * (I_mov > threshold_value)  # konwersja typu logicznego na liczbowy
        I_mov_B = I_mov_B * 255     # zamiana zakresu z {0, 1} na {0, 255} (poprawne wyświetlanie)
        I_mov_B = I_mov_B.astype(np.uint8)

        I_GT = 1 * (I_GT == 255)  # konwersja typu logicznego na liczbowy
        I_GT = I_GT * 255     # zamiana zakresu z {0, 1} na {0, 255} (poprawne wyświetlanie)
        I_GT = I_GT.astype(np.uint8)

        I_mov_B = cv.medianBlur(I_mov_B, 9)
        I_mov_B = cv.erode(I_mov_B, kernel, iterations=3)
        I_mov_B = cv.dilate(I_mov_B, kernel, iterations=3)

        retval, labels, stats, centroids = cv.connectedComponentsWithStats(I_mov_B)

        cv.imshow("Labels", np.uint8(labels / retval * 255))
        cv.imshow('I', I_mov_B)
        cv.imshow('ideal', I_GT)

        I_VIS = I.copy()
        if stats.shape[0] > 1:
            tab = stats[1:, 4]
            pi = np.argmax(tab)
            pi = pi + 1
            cv.rectangle(I_VIS, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0]+stats[pi, 2], stats[pi, 1]+stats[pi, 3]), (255, 0, 0), 2)
            cv.putText(I_VIS, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            cv.putText(I_VIS, "%d" %pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

        cv.imshow("frames", I_VIS)

        TP_M = np.logical_and((I_mov_B == 255), (I_GT == 255))
        TP_S = np.sum(TP_M)
        TP = TP + TP_S

        TN_M = np.logical_and((I_mov_B == 0), (I_GT == 0))
        TN_S = np.sum(TN_M)
        TN = TN + TN_S

        FP_M = np.logical_and((I_mov_B == 255), (I_GT == 0))
        FP_S = np.sum(FP_M)
        FP = FP + FP_S

        FN_M = np.logical_and((I_mov_B == 0), (I_GT == 255))
        FN_S = np.sum(FN_M)
        FN = FN + FN_S

        cv.waitKey(10)
        I_prev = I

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2*P*R) / (P + R)

    print(img_source)
    print('Precyzja: ', P)
    print('Czułość: ', R)
    print('F1: ', F1)
    print()