import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show_img(window_name, img, ascii_key_to_exit=13):
    while True:
        cv.imshow(window_name, np.uint8(img))
        key = cv.waitKey(1) & 0xFF
        if key == ascii_key_to_exit:
            cv.destroyAllWindows()
            break


def hist(img):
    h = np.zeros((256, 1), np.float32)
    img_height, img_width = img.shape[:2]

    for j in range(img_height):
        for k in range(img_width):
            value = img[j][k]
            h[value] += 1
    return h


escape_key_ascii = 13
mandril_img = cv.imread('mandril.jpg',)

show_img("Mandril image", mandril_img, 13)

cv.imwrite('m.png', mandril_img)

print("Rozmiar: ", mandril_img.shape)
print("Liczba bajtów: ", mandril_img.size)
print("Typ danych: ", mandril_img.dtype)

mandril_img_gray = cv.cvtColor(mandril_img, cv.COLOR_BGR2GRAY)
show_img('Mandril gray image', mandril_img_gray, escape_key_ascii)

mandril_img_HSV = cv.cvtColor(mandril_img, cv.COLOR_BGR2HSV)
show_img('Mandril HSV image', mandril_img_HSV, escape_key_ascii)

for i, channel in zip(range(3), ['Hue', 'Saturation', 'Value']):
    mandril_img_HSV_channel = mandril_img_HSV[:, :, i]
    show_img(channel, mandril_img_HSV_channel, escape_key_ascii)

height, width = mandril_img.shape[:2]
scale = 1.75

mandril_img_x2 = cv.resize(mandril_img, (int(scale*height), int(scale*width)))
show_img('Mandril resized image', mandril_img_x2, escape_key_ascii)

lena_img = cv.imread('lena.png')
lena_img_gray = cv.cvtColor(lena_img, cv.COLOR_BGR2GRAY)

img_add = lena_img_gray + mandril_img_gray
show_img("Added images", img_add, escape_key_ascii)

img_sub = lena_img_gray - mandril_img_gray
show_img('Subtracted images', img_sub, escape_key_ascii)

img_mul = lena_img_gray * mandril_img_gray
show_img('Multiplicated images', img_mul, escape_key_ascii)

img_lin_com = 3*lena_img_gray + 2*mandril_img_gray
show_img("Linear combination of images", img_lin_com, escape_key_ascii)

img_abs_sub = abs(lena_img_gray - mandril_img_gray)
show_img("Absolutely subtracted images", img_abs_sub, escape_key_ascii)

mandril_histogram = hist(mandril_img_gray)
mandril_histogram_opencv = cv.calcHist([mandril_img_gray], [0], None, [256], [0, 256])
pixel_values = np.arange(0, 256, 1)

ax1 = plt.subplot(221)
ax1.set_title('Mandril image histogram')
ax1.plot(pixel_values, mandril_histogram)

ax2 = plt.subplot(222)
ax2.set_title('Mandril image OpenCV histogram')
ax2.plot(pixel_values, mandril_histogram_opencv)

ax3 = plt.subplot(212)
ax3.plot(pixel_values, mandril_histogram, label='User histogram')
ax3.plot(pixel_values, mandril_histogram_opencv, label='OpenCV histogram')
ax3.legend()
plt.show(block=True)

mandril_img_gray_equ = cv.equalizeHist(mandril_img_gray)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
mandril_img_gray_cla = clahe.apply(mandril_img_gray)

mandril_img_gray_equ_histogram = cv.calcHist([mandril_img_gray_equ], [0], None, [256], [0, 256])
mandril_img_gray_cla_histogram = cv.calcHist([mandril_img_gray_cla], [0], None, [256], [0, 256])

ax1 = plt.subplot(221)
ax1.plot(pixel_values, mandril_img_gray_equ_histogram)
ax1.set_title('Histogram')

ax2 = plt.subplot(222)
ax2.plot(pixel_values, mandril_img_gray_cla_histogram)
ax2.set_title('Histogram')

ax3 = plt.subplot(212)
ax3.plot(pixel_values, mandril_img_gray_equ_histogram, label='Classic')
ax3.plot(pixel_values, mandril_img_gray_cla_histogram, label='CLAHE')
ax3.legend()
plt.show(block=True)


