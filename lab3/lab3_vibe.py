import numpy as np
import cv2 as cv


def initial_background(I_G, N):
    YY = I_G.shape[0]
    XX = I_G.shape[1]
    samples = np.zeros((YY, XX, N))
    for i in range(1, YY - 1):
        for j in range(1, XX - 1):
            for n in range(N):
                x, y = 0, 0
                while (x == 0 and y == 0):
                    x = np.random.randint(-1, 1)
                    y = np.random.randint(-1, 1)
                ri = i + x
                rj = j + y
                samples[i, j, n] = I_G[ri, rj]
    return samples


def vibe_detection(I_G, samples, _min, fi, N, R):
    YY = I_G.shape[0]
    XX = I_G.shape[1]
    segMap = np.zeros((YY, XX), dtype=np.uint8)
    for i in range(YY):
        for j in range(XX):
            count, index, dist = 0, 0, 0
            while count < _min and index < N:
                dist = np.abs(I_G[i, j] - samples[i, j, index])
                if dist < R:
                    count += 1
                index += 1
            if count >= _min:
                r = np.random.randint(0, fi - 1)
                if r == 0:
                    r = np.random.randint(0, N - 1)
                    samples[i, j, r] = I_G[i, j]
                r = np.random.randint(0, fi - 1)
                if r == 0:
                    x, y = 0, 0
                    while (x == 0 and y == 0):
                        x = np.random.randint(-1, 1)
                        y = np.random.randint(-1, 1)
                    r = np.random.randint(0, N - 1)
                    ri = i + x
                    rj = j + y
                    try:
                        samples[ri, rj, r] = I_G[i, j]
                    except:
                        pass
            else:
                segMap[i, j] = 255
    return segMap, samples

N = 20
R = 20
_min = 2
fi = 16

kernel = np.ones((3, 3), np.int)

for img_source in ['highway', 'pedestrants', 'office']:

    # zmienne do pomiaru dokładności algorytmu
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    f = open(img_source + '/temporalROI.txt', 'r')  # otwarcie pliku
    line = f.readline()  # odczyt lini
    roi_start, roi_end = line.split()  # rozbicie lini na poszczegolne

    roi_start = int(roi_start)
    roi_end = int(roi_end)
    step_frame = 15

    I = cv.imread(img_source + '/input/in%06d.jpg' % roi_start)
    I_G = cv.cvtColor(I, cv.COLOR_BGR2GRAY)

    samples = initial_background(I_G, N)

    for i in range(roi_start, roi_end, step_frame):
        I = cv.imread(img_source + '/input/in%06d.jpg' % i)

        I_GT = cv.imread(img_source + '/groundtruth/gt%06d.png' % i)
        I_GT = cv.cvtColor(I_GT, cv.COLOR_BGR2GRAY)
        I_GT = 1 * (I_GT == 255)  # konwersja typu logicznego na liczbowy
        I_GT = I_GT * 255  # zamiana zakresu z {0, 1} na {0, 255} (poprawne wyświetlanie)
        I_GT = I_GT.astype(np.uint8)  # rzutowanie zmiennych na typ całkowity uint8

        I_G = cv.cvtColor(I, cv.COLOR_BGR2GRAY)

        segMap, samples = vibe_detection(I_G, samples, _min, fi, N, R)

        segMap = cv.medianBlur(segMap, 3)
        # operacje morfologiczne na analizowanym obrazie
        segMap = cv.erode(segMap, kernel, iterations=1)
        segMap = cv.dilate(segMap, kernel, iterations=1)

        cv.imshow('segMap', segMap)

        TP_M = np.logical_and((segMap == 255), (I_GT == 255))
        TP_S = np.sum(TP_M)
        TP = TP + TP_S

        TN_M = np.logical_and((segMap == 0), (I_GT == 0))
        TN_S = np.sum(TN_M)
        TN = TN + TN_S

        FP_M = np.logical_and((segMap == 255), (I_GT == 0))
        FP_S = np.sum(FP_M)
        FP = FP + FP_S

        FN_M = np.logical_and((segMap == 0), (I_GT == 255))
        FN_S = np.sum(FN_M)
        FN = FN + FN_S

        cv.waitKey(1)

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)

    print(img_source)
    print('Precyzja: ', P)
    print('Czułość: ', R)
    print('F1: ', F1)
    print()

    cv.destroyAllWindows()