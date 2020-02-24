import cv2 as cv
import numpy as np

def hist(img):
    h = np.zeros((256,1), np.float32)
    hei


I = cv.imread('mandril.jpg',)

cv.imshow("image", I)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('m.png', I)

print("Rozmiar: ", I.shape)
print("Liczba bajtów: ", I.size)
print("Typ danych: ", I.dtype)


#Obrz w skali szarości
IG = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
cv.imshow("image gray", IG)
cv.waitKey(0)
cv.destroyAllWindows()

#Obraz w przestrzeni barw HSV
IHSV = cv.cvtColor(I, cv.COLOR_BGR2HSV)
cv.imshow("image hsv", IHSV)
cv.waitKey(0)
cv.destroyAllWindows()

for i, channel in zip(range(3), ['hue', 'saturation', 'value']):
    IHSV_channel = IHSV[:,:,i]
    cv.imshow(channel, IHSV_channel)
    cv.waitKey(0)
    cv.destroyAllWindows()

height, width = I.shape[:2]

scale = 1.75
Ix2 = cv.resize(I, (int(scale*height), int(scale*width)))
cv.imshow("big image", Ix2)
cv.waitKey(0)
cv.destroyAllWindows()

I2 = cv.imread('lena.png')
I2G = cv.cvtColor(I2, cv.COLOR_BGR2GRAY)

Iadd = I2G + IG
cv.imshow('added images', Iadd)
cv.waitKey(0)
cv.destroyAllWindows()

Idiff = I2G - IG
cv.imshow('difference between images', Idiff)
cv.waitKey(0)
cv.destroyAllWindows()

Imul = I2G * IG
cv.imshow('multiplicated images', Imul)
cv.waitKey(0)
cv.destroyAllWindows()

