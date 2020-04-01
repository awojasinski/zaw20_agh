import cv2 as cv
import numpy as np


cap = cv.VideoCapture('vid1_IR.avi')

DPM = cv.imread('dmp.png')
DPM = cv.cvtColor(DPM, cv.COLOR_BGR2GRAY)
DPM = DPM.astype(np.float32)
DPM_1 = DPM/float(50)
DPM_0 = 1 - DPM_1

sampleSize = (64, 192)
stepSize = 16
threshVal = 45

while (cap.isOpened()):
    ret, frame = cap.read()
    I_G = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    B = cv.threshold(I_G, threshVal, 255, cv.THRESH_TOZERO)[1]

    # Okno przesuwne
    result = np.zeros((360, 480), dtype=np.float32)

    for y in range(0, frame.shape[0], stepSize):
        if y + sampleSize[1] >= frame.shape[0]:
            win_height = frame.shape[0]
            mask_height = sampleSize[1] - ((y + sampleSize[1]) - frame.shape[0])
        else:
            win_height = sampleSize[1] + y
            mask_height = sampleSize[1]
        p_y = y + int(sampleSize[1] / 2) if y + int(sampleSize[1] / 2) < frame.shape[0] else y
        for x in range(0, frame.shape[1], stepSize):
            if x + sampleSize[0] >= frame.shape[1]:
                win_width = frame.shape[1]
                mask_width = sampleSize[0] - ((x + sampleSize[0]) - frame.shape[1])
            else:
                win_width = sampleSize[0] + x
                mask_width = sampleSize[0]
            B_tmp = B[y:win_height, x:win_width]
            temp = sum(sum(B_tmp * DPM_1[0:mask_height, 0:mask_width] + np.subtract(
                np.ones(B_tmp.shape, dtype=np.float32), B_tmp) * DPM_0[0:mask_height, 0:mask_width]))

            p_x = x + int(sampleSize[0] / 2) if x + int(sampleSize[0] / 2) < frame.shape[1] else x

            result[p_y, p_x] = temp

    result = result / np.max(result)
    int8 = np.uint8(result * 255)

    while True:
        ind = np.unravel_index(np.argmax(result, axis=None), result.shape) # Szukanie maksimum lokalnego
        y_max, x_max = ind
        if (result[y_max, x_max] < 0.5): 
            break
        # Wyrysowanie ramki wokół maksimum
        rect_x = x_max - int(sampleSize[0]/2)
        rect_y = y_max - int(sampleSize[1]/2)
        result[rect_y:rect_y+sampleSize[1], rect_x:rect_x+sampleSize[0]] = 0 # Wyzerowanie użytych pikseli
        # Wyrysowanie prostokąta
        cv.rectangle(frame, (rect_x, rect_y), (rect_x+sampleSize[0], rect_y+sampleSize[1]), (255, 0, 0), 2)

    cv.imshow('Obraz z kamery', frame)
    cv.imshow("Labels", int8)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

"""
Jedną z możliwości detekcji obiektów w innym rozmiarze mogłoby być poszukiwanie wzorca
w otoczeniu znalezionego już, tzn. gdy wykryjemy na obrazie wzorce nakładamy maskę i
szukamy wzorca w obrębie obrazu z nałożoną maską
"""