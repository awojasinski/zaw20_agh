import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib


def show_img(window_name, img, ascii_key_to_exit=13, save_img=False, img_name=''):
    if save_img:
        cv.imwrite(img_name, img)
    while True:
        cv.imshow(window_name, np.uint8(img))
        key = cv.waitKey(1) & 0xFF
        if key == ascii_key_to_exit:
            cv.destroyAllWindows()
            break


def add_img_title(text, img, bgr_background=(0, 0, 0), bgr_text=(255, 255, 255)):

    imgB = np.full(shape=(int(img.shape[0] / 8), img.shape[1]), fill_value=bgr_background[0])
    imgG = np.full(shape=(int(img.shape[0] / 8), img.shape[1]), fill_value=bgr_background[1])
    imgR = np.full(shape=(int(img.shape[0] / 8), img.shape[1]), fill_value=bgr_background[2])
    title_img = np.dstack((imgB, imgG, imgR))

    title_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    titleX = int((title_img.shape[1] - title_size[0]) / 2)
    titleY = int((title_img.shape[0] + title_size[1]) / 2)
    title_img = cv.putText(title_img, text, (titleX, titleY), cv.FONT_HERSHEY_SIMPLEX, 1, bgr_text, 2)
    img = np.vstack((title_img, img))
    return img


def hist(img):
    h = np.zeros((256, 1), np.float32)
    img_height, img_width = img.shape[:2]

    for j in range(img_height):
        for k in range(img_width):
            value = img[j][k]
            h[value] += 1
    return h


def rgb2gray(I):
    return 0.299*I[:, :, 0] + 0.587*I[:, :, 1] + 0.144*I[:, :, 2]


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
ax1.set_xlim(0, 255)

ax2 = plt.subplot(222)
ax2.set_title('Mandril image OpenCV histogram')
ax2.plot(pixel_values, mandril_histogram_opencv)
ax2.set_xlim(0, 255)

ax3 = plt.subplot(212)
ax3.plot(pixel_values, mandril_histogram, label='User histogram')
ax3.plot(pixel_values, mandril_histogram_opencv, label='OpenCV histogram')
ax3.legend()
ax3.set_xlim(0, 255)
plt.show(block=True)

mandril_img_gray_equ = cv.equalizeHist(mandril_img_gray)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
mandril_img_gray_cla = clahe.apply(mandril_img_gray)

mandril_img_gray_equ_histogram = cv.calcHist([mandril_img_gray_equ], [0], None, [256], [0, 256])
mandril_img_gray_cla_histogram = cv.calcHist([mandril_img_gray_cla], [0], None, [256], [0, 256])

ax1 = plt.subplot(221)
ax1.plot(pixel_values, mandril_img_gray_equ_histogram)
ax1.set_title('Classic histogram')
ax1.set_xlim(0, 255)

ax2 = plt.subplot(222)
ax2.plot(pixel_values, mandril_img_gray_cla_histogram)
ax2.set_title('CLAHE histogram')
ax2.set_xlim(0, 255)

ax3 = plt.subplot(212)
ax3.set_xlim(0, 255)
ax3.plot(pixel_values, mandril_img_gray_equ_histogram, label='Classic')
ax3.plot(pixel_values, mandril_img_gray_cla_histogram, label='CLAHE')
ax3.legend()
plt.show(block=True)

mandril_img_gray = cv.cvtColor(mandril_img_gray, cv.COLOR_GRAY2BGR)
mandril_img_gray_equ = cv.cvtColor(mandril_img_gray_equ, cv.COLOR_GRAY2BGR)
mandril_img_gray_cla = cv.cvtColor(mandril_img_gray_cla, cv.COLOR_GRAY2BGR)
mandril_img_gray_text = add_img_title('Gray image', mandril_img_gray)
mandril_img_gray_equ_text = add_img_title('Classic histogram equalization', mandril_img_gray_equ)
mandril_img_gray_cla_text = add_img_title('CLAHE histogram equalization', mandril_img_gray_cla)
compared_histogram_equ_img = np.hstack((mandril_img_gray_text, mandril_img_gray_equ_text, mandril_img_gray_cla_text))
show_img('Gray vs Classic vs CLAHE', compared_histogram_equ_img, 13)

mandril_img = cv.cvtColor(mandril_img, cv.COLOR_BGR2RGB)

ax1 = plt.subplot(221)
ax1.imshow(mandril_img)
ax1.set_title('Original')
ax1.axis('off')

ax2 = plt.subplot(222)
ax2.imshow(cv.GaussianBlur(mandril_img, (5, 5), 0))
ax2.set_title('Gaussian blur 5x5')
ax2.axis('off')

ax3 = plt.subplot(223)
ax3.imshow(cv.GaussianBlur(mandril_img, (9, 9), 0))
ax3.set_title('Gaussian blur 9x9')
ax3.axis('off')

ax4 = plt.subplot(224)
ax4.imshow(cv.GaussianBlur(mandril_img, (13, 13), 0))
ax4.set_title('Gaussian blur 13x13')
ax4.axis('off')

plt.show(block=True)

mandril_img_gray_gaussian = cv.GaussianBlur(mandril_img_gray, (3,3), 0)

mandril_img_grad_x = cv.Sobel(mandril_img_gray_gaussian, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
mandril_img_grad_y = cv.Sobel(mandril_img_gray_gaussian, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

mandril_img_grad_x_abs = abs(mandril_img_grad_x)
mandril_img_grad_y_abs = abs(mandril_img_grad_y)

mandril_img_grad = cv.addWeighted(mandril_img_grad_x_abs, 0.5, mandril_img_grad_y_abs, 0.5, 0)

ax1 = plt.subplot(221)
ax1.imshow(mandril_img_gray)
ax1.set_title('Original gray')
ax1.axis('off')

ax2 = plt.subplot(222)
ax2.imshow(mandril_img_grad)
ax2.set_title('Sobel')
ax2.axis('off')

ax3 = plt.subplot(223)
ax3.imshow(mandril_img_grad_x_abs)
ax3.set_title('Sobel X')
ax3.axis('off')

ax4 = plt.subplot(224)
ax4.imshow(mandril_img_grad_y_abs)
ax4.set_title('Sobel Y')
ax4.axis('off')

plt.show(block=True)

ax1 = plt.subplot(121)
ax1.imshow(mandril_img_grad)
ax1.set_title('Sobel')
ax1.axis('off')

ax2 = plt.subplot(122)
ax2.imshow(cv.Laplacian(mandril_img_gray_gaussian, cv.CV_16S, ksize=3))
ax2.set_title('Laplacian')
ax2.axis('off')

plt.show(block=True)

ax1 = plt.subplot(221)
ax1.imshow(mandril_img)
ax1.set_title('Original')
ax1.axis('off')

ax2 = plt.subplot(222)
ax2.imshow(cv.medianBlur(mandril_img, 5))
ax2.set_title('Median blur 5x5')
ax2.axis('off')

ax3 = plt.subplot(223)
ax3.imshow(cv.medianBlur(mandril_img, 9))
ax3.set_title('Median blur 9x9')
ax3.axis('off')

ax4 = plt.subplot(224)
ax4.imshow(cv.medianBlur(mandril_img, 13))
ax4.set_title('Median blur 13x13')
ax4.axis('off')

plt.show(block=True)


I = plt.imread('mandril.jpg')

fig, ax = plt.subplots(1)
plt.imshow(I)
plt.title('Mandril')
plt.axis('off')

plt.imsave('mandril.jpg', I)

x = [100, 150, 200, 250]
y = [50, 100, 150, 200]
plt.plot(x, y, 'r.', markersize=10)

rect = Rectangle((50, 50), 50, 100, fill=False, ec='r')
ax.add_patch(rect)
plt.show(block=True)

plt.figure(1)
IG = rgb2gray(I)
plt.imshow(IG)
plt.title('Mandril gray')
plt.axis('off')
plt.gray()
plt.show(block=True)

plt.figure(1)
_HSV = matplotlib.colors.rgb_to_hsv(I)
plt.imshow(_HSV)
plt.title('Mandril HSV')
plt.axis('off')
plt.show(block=True)
