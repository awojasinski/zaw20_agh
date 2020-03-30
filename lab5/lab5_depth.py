import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_L = cv.imread('aloes/aloeL.jpg', 0)
img_R = cv.imread('aloes/aloeR.jpg', 0)
GT = cv.imread('aloes/gt.png', 0)


minDis = 16
numDis = 96
blockS = 17

stereoBM = cv.StereoBM_create(numDisparities=numDis, blockSize=blockS)
stereoSGBM = cv.StereoSGBM_create(minDisparity=minDis, numDisparities=numDis,
                                  blockSize=blockS, disp12MaxDiff=0,
                                  uniquenessRatio=10, speckleWindowSize=100,
                                  speckleRange=32)

stereoBM.setMinDisparity(minDis)
stereoBM.setDisp12MaxDiff(0)
stereoBM.setUniquenessRatio(10)
stereoBM.setSpeckleRange(32)
stereoBM.setSpeckleWindowSize(100)

img_depthBM = stereoBM.compute(img_L, img_R)
img_depthBM = (img_depthBM / np.max(img_depthBM)) * 255

img_depthSGBM = stereoSGBM.compute(img_L, img_R)
img_depthSGBM = (img_depthSGBM / np.max(img_depthSGBM)) * 255

plt.subplot(131)
plt.title('Block matching')
plt.imshow(img_depthBM, cmap='gray')
plt.axis('off')
plt.subplot(132)
plt.title('Semi-Global Matching')
plt.imshow(img_depthSGBM, cmap='gray')
plt.axis('off')
plt.subplot(133)
plt.title('Groudtruth')
plt.imshow(GT, cmap='gray')
plt.axis('off')
plt.show()

rms_bm = np.sqrt(np.sum(np.power(img_depthBM - GT, 2)) / (GT.shape[0]*GT.shape[1]))
rms_sgbm = np.sqrt(np.sum(np.power(img_depthSGBM - GT, 2)) / (GT.shape[0]*GT.shape[1]))

print('Wskaznik rms dla metody Block matching wynosi:', rms_bm)
print('Wskaznik rms dla metody Semi-Global Matching wynosi:', rms_sgbm)
