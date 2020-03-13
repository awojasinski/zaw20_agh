import cv2 as cv
import numpy as np

# iteracja po każdym zbiorze obrazów
for img_source in ['highway', 'office', 'pedestrants']:
    N = 60  # rozmiar bufora

    kernel = np.ones((2, 2), np.int)
    threshold_value = 10

    f = open(img_source + '/temporalROI.txt', 'r')   # otwarcie pliku
    line = f.readline()     # odczyt lini
    roi_start, roi_end = line.split()   # rozbicie lini na poszczegolne

    roi_start = int(roi_start)
    roi_end = int(roi_end)
    step_frame = 1

    # sprawdzenie rozdzielczości obrazu
    I = cv.imread(img_source + '/input/in%06d.jpg' % roi_start)
    YY = I.shape[0]     # wysokość
    XX = I.shape[1]     # szerokość

    I_model = np.zeros((YY, XX), np.uint8)
    BUF = np.zeros((YY, XX, N), np.uint8)
    iN = 0

    # dwie metody obliczania modelu tła
    for method in ['mean', 'median']:
        # zmienne do pomiaru dokładności algorytmu
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # utworzenie buforu zdjęć potrzebne do stworzenia modelu
        # zdjęcia przed pojawieniem się ruchu na obrazie
        for i in range(roi_start-N, roi_start, step_frame):
            I = cv.imread(img_source + '/input/in%06d.jpg' % i)
            I_G = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
            BUF[:, :, iN] = I_G

            iN += 1
            if iN == N:
                iN = 0

        for i in range(roi_start, roi_end, step_frame):

            I = cv.imread(img_source + '/input/in%06d.jpg' % i)
            I_GT = cv.imread(img_source + '/groundtruth/gt%06d.png' % i)

            I_G = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
            I_GT = cv.cvtColor(I_GT, cv.COLOR_BGR2GRAY)
            BUF[:, :, iN] = I_G

            iN += 1
            if iN == N:
                iN = 0

            if method == 'median':
                I_model = np.median(BUF, axis=2)
                I_mov = cv.absdiff(I_G, I_model)
            elif method == 'mean':
                I_model = np.mean(BUF, axis=2)
                I_mov = abs(I_G - I_model)

            I_B = 1 * (I_mov > threshold_value)  # konwersja typu logicznego na liczbowy
            I_B = I_B * 255  # zamiana zakresu z {0, 1} na {0, 255} (poprawne wyświetlanie)
            I_B = I_B.astype(np.uint8)  # rzutowanie zmiennych na typ całkowity uint8

            I_GT = 1 * (I_GT == 255)  # konwersja typu logicznego na liczbowy
            I_GT = I_GT * 255  # zamiana zakresu z {0, 1} na {0, 255} (poprawne wyświetlanie)
            I_GT = I_GT.astype(np.uint8)  # rzutowanie zmiennych na typ całkowity uint8

            I_B = cv.medianBlur(I_B, 7)
            # operacje morfologiczne na analizowanym obrazie
            I_B = cv.erode(I_B, kernel, iterations=3)
            I_B = cv.dilate(I_B, kernel, iterations=2)

            retval, labels, stats, centroids = cv.connectedComponentsWithStats(I_B)

            # rysowanie prostokąta wokół największego ruchomego obiektu
            I_VIS = I.copy()
            if stats.shape[0] > 1:
                tab = stats[1:, 4]
                pi = np.argmax(tab)
                pi = pi + 1
                cv.rectangle(I_VIS, (stats[pi, 0], stats[pi, 1]),
                             (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]), (255, 0, 0), 2)
                cv.putText(I_VIS, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 0, 0))
                cv.putText(I_VIS, "%d" % pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])), cv.FONT_HERSHEY_SIMPLEX,
                           1, (255, 0, 0))

            # połączenie trzech obrazów w jeden (łatwiejsza analiza przez użytkownika, mniejszy bałagan)
            labels_img = cv.cvtColor(np.uint8(labels / retval * 255), cv.COLOR_GRAY2BGR)
            I_GT_bgr = cv.cvtColor(I_GT, cv.COLOR_GRAY2BGR)
            video = cv.hconcat([labels_img, I_VIS, I_GT_bgr])
            cv.imshow('Video: %s method' % (method), video)

            TP_M = np.logical_and((I_B == 255), (I_GT == 255))
            TP_S = np.sum(TP_M)
            TP = TP + TP_S

            TN_M = np.logical_and((I_B == 0), (I_GT == 0))
            TN_S = np.sum(TN_M)
            TN = TN + TN_S

            FP_M = np.logical_and((I_B == 255), (I_GT == 0))
            FP_S = np.sum(FP_M)
            FP = FP + FP_S

            FN_M = np.logical_and((I_B == 0), (I_GT == 255))
            FN_S = np.sum(FN_M)
            FN = FN + FN_S

            cv.waitKey(10)
            I_prev = I

        cv.destroyAllWindows()

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = (2 * P * R) / (P + R)

        print(img_source)
        print('Metoda: ', method)
        print('Precyzja: ', P)
        print('Czułość: ', R)
        print('F1: ', F1)
        print()


