import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# wczytanie obrazów
I = cv.imread('I.jpg')
J = cv.imread('J.jpg')

# konwersja przestrzeni barw
I = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
J = cv.cvtColor(J, cv.COLOR_BGR2GRAY)
I_sub = cv.absdiff(I, J)    # odjęcie obrazó

# wyświetlenie obrazów
cv.imshow('I', I)
cv.imshow('J', J)
cv.imshow('differnece', I_sub)
cv.waitKey(0)

W2 = 1
dX = 1
dY = 1

height, width = I.shape[0:2]    # odczytanie rozdzielczonści obrazu

# inicjalizacja macierzy do przepływu optycznego
u = np.zeros((height, width))
v = np.zeros((height, width))

for j in range(W2+1, height-W2-1):
    for i in range(W2+1, width-W2-1):
        IO = np.float32(I[j-W2:j+W2+1, i-W2:i+W2+1])
        min_dist = 10000000
        for j_1 in range(j-dY, j+dY+1):
            for i_1 in range(i - dX, i + dX + 1):
                JO = np.float32(J[j_1-W2:j_1+W2+1, i_1-W2:i_1+W2+1])
                dist = np.sum(np.sqrt((np.square(JO - IO))))
                if dist < min_dist:
                    min_dist = dist
                    u[j, i] = j_1 - j
                    v[j, i] = i_1 - i
# wykres przepływu optycznego
plt.quiver(u, v)
plt.gca().invert_yaxis()
plt.show()
