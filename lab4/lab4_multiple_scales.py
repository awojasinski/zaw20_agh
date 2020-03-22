import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def of(I, J, u0, v0, W2=1, dY=1, dX=1):

    height, width = I.shape[0:2]  # odczytanie rozdzielczonści obrazu

    # inicjalizacja macierzy do przepływu optycznego
    u = np.zeros((height, width))
    v = np.zeros((height, width))

    for j in range(W2 + 1, height - W2 - 1):
        for i in range(W2 + 1, width - W2 - 1):
            IO = np.float32(I[j - W2:j + W2 + 1, i - W2:i + W2 + 1])
            min_dist = 10000000
            for j_1 in range(j - dY, j + dY + 1):
                for i_1 in range(i - dX, i + dX + 1):
                    if j_1 < (height-W2) and i_1 < (width-W2) and i_1 > W2 and j_1 > W2:
                        JO = np.float32(J[j_1 - int(u0[j, i]) - W2:j_1 - int(u0[j, i]) + W2 + 1,
                                       i_1 - int(v0[j, i]) - W2:i_1 - int(v0[j, i]) + W2 + 1])
                        dist = np.sum(np.sqrt((np.square(JO - IO))))
                        if dist < min_dist:
                            min_dist = dist
                            u[j, i] = j_1 - j - u0[j, i]
                            v[j, i] = i_1 - i - v0[j, i]
    return u, v


def pyramid(im, max_scale):
    images = [im]
    for k in range(1, max_scale):
        images.append(cv.resize(images[k-1], (0, 0), fx=0.5, fy=0.5))
    return images


# wczytanie obrazów
I = cv.imread('data/I.jpg')
J = cv.imread('data/J.jpg')

# konwersja przestrzeni barw
I = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
J = cv.cvtColor(J, cv.COLOR_BGR2GRAY)
I_sub = cv.absdiff(I, J)    # odjęcie obrazó

# wyświetlenie obrazów
cv.imshow('I', I)
cv.imshow('J', J)
cv.imshow('differnece', I_sub)
cv.waitKey(0)

K = 3

IP = pyramid(I, K)
JP = pyramid(J, K)
u0 = np.zeros(IP[-1].shape, np.float32)
v0 = np.zeros(JP[-1].shape, np.float32)

u, v = of(IP[-1], JP[-1], u0, v0)
for k in range(1, K):
    v = cv.resize(v, (0, 0), fx=2, fy=2, interpolation=cv.INTER_NEAREST)
    u = cv.resize(u, (0, 0), fx=2, fy=2, interpolation=cv.INTER_NEAREST)
    u, v = of(IP[-k-1], JP[-k-1], u, v)

# wykres przepływu optycznego
plt.quiver(u, v)
plt.gca().invert_yaxis()
plt.show()
