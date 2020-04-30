import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h, h))


def highpassFilter(size):
    rows = np.cos(np.pi*np.array([-0.5 + x / (size[0] - 1) for x in range(size[0])]))
    cols = np.cos(np.pi*np.array([-0.5 + x / (size[1] - 1) for x in range(size[1])]))
    X = np.outer(rows, cols)
    return (1.0 - X) * (2.0 - X)


def img_preprocessing(img):
    im = cv.imread(img)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    return im


# Wczytanie wzoru
im_w_in = img_preprocessing('obrazy_Mellin/domek_r0_64.pgm')

# Przeszukiwanie obrazów obróconych
for i in range(0, 330, 30):
    img = 'obrazy_Mellin/domek_r' + str(i) + '.pgm'
    im_p = img_preprocessing(img)

    im_w2 = np.zeros(im_p.shape)
    im_w = im_w_in * hanning2D(im_w_in.shape[0])
    im_w2[0:im_w.shape[0], 0:im_w.shape[1]] = im_w

    im_p_fft = np.fft.fft2(im_p)
    im_p_fft = np.fft.fftshift(im_p_fft)
    im_w2_fft = np.fft.fft2(im_w2)
    im_w2_fft = np.fft.fftshift(im_w2_fft)

    im_p_f = np.abs(im_p_fft) * highpassFilter(im_p_fft.shape)
    im_w2_f = np.abs(im_w2_fft) * highpassFilter(im_w2_fft.shape)

    im_p_fft = np.abs(im_p_f)
    im_w2_fft = np.abs(im_w2_f)

    M = im_w2_fft.shape[0] / np.log(im_w2_fft.shape[0] // 2)
    center = (im_w2_fft.shape[0] // 2, im_w2_fft.shape[1] // 2)
    im_w2_logpolar = cv.logPolar(im_w2_fft, center, M, cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS)
    im_p_logpolar = cv.logPolar(im_p_fft, center, M, cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS)
    im_p_fft = np.fft.fft2(im_p_logpolar)
    im_w2_fft = np.fft.fft2(im_w2_logpolar)

    conj = np.conj(im_w2_fft) * im_p_fft
    conj = conj / np.abs(conj)
    corr = np.abs(np.fft.ifft2(conj))

    alpha_p, logr_p = np.unravel_index(np.argmax(corr), corr.shape)
    logr_size = im_p_logpolar.shape[0]
    alpha_size = im_p_logpolar.shape[1]

    if logr_p > logr_size // 2:
        w = logr_size - logr_p  # powiekszenie
    else:
        w = - logr_p  # pomniejszenie

    A = (alpha_p * 360.0) / alpha_size
    a1 = - A
    a2 = 180 - A

    scale = np.exp(w / M)  # M to parametr funkcji cv2.logPolar
    print(scale)

    im = np.zeros(im_p.shape)
    x_1 = int((im_p.shape[0] - im_w.shape[0]) / 2)
    x_2 = int((im_p.shape[0] + im_w.shape[0]) / 2)
    y_1 = int((im_p.shape[1] - im_w.shape[1]) / 2)
    y_2 = int((im_p.shape[1] + im_w.shape[1]) / 2)
    im[x_1:x_2, y_1:y_2] = im_w_in

    centerT = (im.shape[0] / 2 - 0.5, im.shape[1] / 2 - 0.5)
    # im to obraz wzorca uzupelniony zerami, ale ze wzorcem umieszczonym na srodku, a nie w lewym, gornym rogu!
    transM1 = cv.getRotationMatrix2D(centerT, a1, scale)
    im_r_s1 = cv.warpAffine(im, transM1, im.shape)

    transM2 = cv.getRotationMatrix2D(centerT, a2, scale)
    im_r_s2 = cv.warpAffine(im, transM2, im.shape)

    im_fft_1 = np.fft.fft2(im_r_s1)
    im_fft_2 = np.fft.fft2(im_r_s2)
    im_p_fft = np.fft.fft2(im_p)

    conj1 = np.conj(im_fft_1) * im_p_fft
    conj1 = conj1 / np.abs(conj1)
    corr_1 = np.abs(np.fft.ifft2(conj1))

    conj2 = np.conj(im_fft_2) * im_p_fft
    conj2 = conj2 / np.abs(conj2)
    corr_2 = np.abs(np.fft.ifft2(conj2))

    if np.amax(corr_1) > np.amax(corr_2):
        corr_m = corr_1
        pattern_m = im_r_s1
    else:
        corr_m = corr_2
        pattern_m = im_r_s2

    dy, dx = np.unravel_index(np.argmax(corr_m), corr_m.shape)
    if dx > im_p.shape[0] - 5:
        dx = dx - im_p.shape[0]

    print(dx, dy)
    transM = np.float32([[1, 0, dx], [0, 1, dy]])  # dx, dy - wektor przesuniecia
    im_m = cv.warpAffine(pattern_m, transM, (im_p.shape[1], im_p.shape[0]))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(im_m, cmap='gray')
    plt.axis('off')
    plt.title('Wzorzec - obrocony i przeskalowany')

    plt.subplot(1, 2, 2)
    plt.imshow(im_p, cmap='gray')
    plt.axis('off')
    plt.title('Obraz przeszukiwany')

for i in range(10, 80, 10):
    img = 'obrazy_Mellin/domek_s' + str(i) + '.pgm'
    im_p = img_preprocessing(img)

    im_w2 = np.zeros(im_p.shape)
    im_w = im_w_in * hanning2D(im_w_in.shape[0])
    im_w2[0:im_w.shape[0], 0:im_w.shape[1]] = im_w

    im_p_fft = np.fft.fft2(im_p)
    im_p_fft = np.fft.fftshift(im_p_fft)
    im_w2_fft = np.fft.fft2(im_w2)
    im_w2_fft = np.fft.fftshift(im_w2_fft)

    im_p_f = np.abs(im_p_fft) * highpassFilter(im_p_fft.shape)
    im_w2_f = np.abs(im_w2_fft) * highpassFilter(im_w2_fft.shape)

    im_p_fft = np.abs(im_p_f)
    im_w2_fft = np.abs(im_w2_f)

    M = im_w2_fft.shape[0] / np.log(im_w2_fft.shape[0] // 2)
    center = (im_w2_fft.shape[0] // 2, im_w2_fft.shape[1] // 2)
    im_w2_logpolar = cv.logPolar(im_w2_fft, center, M, cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS)
    im_p_logpolar = cv.logPolar(im_p_fft, center, M, cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS)
    im_p_fft = np.fft.fft2(im_p_logpolar)
    im_w2_fft = np.fft.fft2(im_w2_logpolar)
    conj = np.conj(im_w2_fft) * im_p_fft
    conj = conj / np.abs(conj)
    corr = np.abs(np.fft.ifft2(conj))

    alpha_p, logr_p = np.unravel_index(np.argmax(corr), corr.shape)
    logr_size = im_p_logpolar.shape[0]
    alpha_size = im_p_logpolar.shape[1]

    if logr_p > logr_size // 2:
        w = logr_size - logr_p  # powiekszenie
    else:
        w = - logr_p  # pomniejszenie

    A = (alpha_p * 360.0) / alpha_size
    a1 = - A
    a2 = 180 - A

    scale = np.exp(w / M)  # M to parametr funkcji cv2.logPolar
    print(scale)

    im = np.zeros(im_p.shape)
    x_1 = int((im_p.shape[0] - im_w.shape[0]) / 2)
    x_2 = int((im_p.shape[0] + im_w.shape[0]) / 2)
    y_1 = int((im_p.shape[1] - im_w.shape[1]) / 2)
    y_2 = int((im_p.shape[1] + im_w.shape[1]) / 2)
    im[x_1:x_2, y_1:y_2] = im_w_in

    centerT = (im.shape[0] / 2 - 0.5, im.shape[1] / 2 - 0.5)
    # im to obraz wzorca uzupelniony zerami, ale ze wzorcem umieszczonym na srodku, a nie w lewym, gornym rogu!
    transM1 = cv.getRotationMatrix2D(centerT, a1, scale)
    im_r_s1 = cv.warpAffine(im, transM1, im.shape)

    transM2 = cv.getRotationMatrix2D(centerT, a2, scale)
    im_r_s2 = cv.warpAffine(im, transM2, im.shape)

    im_fft_1 = np.fft.fft2(im_r_s1)
    im_fft_2 = np.fft.fft2(im_r_s2)
    im_p_fft = np.fft.fft2(im_p)

    conj1 = np.conj(im_fft_1) * im_p_fft
    conj1 = conj1 / np.abs(conj1)
    corr_1 = np.abs(np.fft.ifft2(conj1))

    conj2 = np.conj(im_fft_2) * im_p_fft
    conj2 = conj2 / np.abs(conj2)
    corr_2 = np.abs(np.fft.ifft2(conj2))

    if np.amax(corr_1) > np.amax(corr_2):
        corr_m = corr_1
        pattern_m = im_r_s1
    else:
        corr_m = corr_2
        pattern_m = im_r_s2

    dy, dx = np.unravel_index(np.argmax(corr_m), corr_m.shape)
    dy = dy - im_p.shape[1]

    print(dx, dy)
    transM = np.float32([[1, 0, dx], [0, 1, dy]])  # dx, dy - wektor przesuniecia
    im_m = cv.warpAffine(pattern_m, transM, (im_p.shape[1], im_p.shape[0]))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(im_m, cmap='gray')
    plt.axis('off')
    plt.title('Wzorzec - obrocony i przeskalowany')

    plt.subplot(1, 2, 2)
    plt.imshow(im_p, cmap='gray')
    plt.axis('off')
    plt.title('Obraz przeszukiwany')

plt.show()
