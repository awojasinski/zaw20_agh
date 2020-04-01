import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#cap = cv.VideoCapture('vid1_IR.avi')
DMP = cv.imread('dmp.png')
DMP = cv.cvtColor(DMP, cv.COLOR_BGR2GRAY)
DMP.astype(np.float32)
DMP_1 = DMP/50      # prawdopodobieństwo zakres 0-1
DMP_0 = 1 - DMP_1   # negacja prawdopodobieństwa
result = np.zeros(shape=(360, 480), dtype=np.float32)

thresholdVal = 45
error_val = 30
kernel = np.ones((2, 2), np.uint8)
r_height = 194
r_width = 64
step = 3


#while cap.isOpened():
#ret, frame = cap.read()
frame = cv.imread('frame_003090.png')
G = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

B = cv.threshold(G, thresholdVal, 1, cv.THRESH_BINARY)[1]
B.astype(np.float32)
for x in range(0, frame.shape[1], step):
    for y in range(0, frame.shape[0], step):
        if y + r_height >= frame.shape[0]:
            h_mask = r_height - ((y + r_height) - frame.shape[0])
        else:
            h_mask = r_height
        if x + r_width >= frame.shape[1]:
            w_mask = r_width - ((x + r_width) - frame.shape[1])
        else:
            w_mask = r_width
        xx = np.sum(B[y:y+h_mask, x:x+w_mask]*DMP_1[0:h_mask, 0:w_mask]+(1-B[y:y+h_mask, x:x+w_mask])*DMP_0[0:h_mask, 0:w_mask])
        result[y, x] = xx

result = result / np.max(np.max(result))
int8 = np.uint8(result*255)
while True:
        ind = np.unravel_index(np.argmax(result, axis=None), result.shape) # Wykrycie maksimum lokalnego
        y_max, x_max = ind
        if (result[y_max, x_max] < 0.5): # Pętla przerwie się, jeśli nie znajdzie się kolejny jasny punkt >= 0.5
            break
        # Wyrysowanie ramki wokół maksimum
        rect_x = x_max - int(r_width/2)
        rect_y = y_max - int(r_height/2)
        result[rect_y:rect_y+r_height, rect_x:rect_x+r_width] = 0 # Wyzerowanie wykorzystanych punktów
        # Wyrysowanie prostokąta
        cv.rectangle(frame, (rect_x, rect_y), (rect_x+r_width, rect_y+r_height), (0, 255, 0), 2)
cv.imshow("i", int8)
cv.imwrite("result.png", int8)
cv.waitKey(0)
