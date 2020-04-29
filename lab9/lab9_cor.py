import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def img_preprocessing(img):
    im = cv.imread(img)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    return im


# Wczytanie obrazów
im_w = img_preprocessing('obrazy_Mellin/wzor.pgm')
im_p = img_preprocessing('obrazy_Mellin/domek_r0.pgm')

# Uzupełnienie wzorca zerami
im_w2 = np.zeros(im_p.shape)
im_w2[0:im_w.shape[0], 0:im_w.shape[1]] = im_w

# Transformaty częstotliwościowe Fouriera
im_w2_fft = np.fft.fft2(im_w2)
im_p_fft = np.fft.fft2(im_p)

comnj = np.conj(im_w2_fft) * im_p_fft
comnj = comnj / np.abs(comnj)
corr = np.abs(np.fft.ifft2(comnj))

y, x = np.unravel_index(np.argmax(corr), corr.shape)
macierz_translacji = np.float32([[1, 0, x], [0, 1, y]])     #x, y - wektor przesuniecia
im_m = cv.warpAffine(im_w, macierz_translacji, (im_p.shape[1], im_p.shape[0]))

# Wyświetlenie obrazów
plt.imshow(im_w, cmap='gray')
plt.title('wzór')
plt.axis('off')
plt.show()

plt.imshow(im_p, cmap='gray')
plt.plot(x, y, '*m')
plt.title('domki')
plt.axis('off')
plt.show()

plt.imshow(im_m, cmap='gray')
plt.title('obraz przesuniety')
plt.axis('off')
plt.show()
