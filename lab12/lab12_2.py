import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from lab12_1 import *
from sklearn import svm

l_prob = 800
HOG_data = np.zeros([2*l_prob, 3781], np.float32)

for i in range(0, l_prob):
    IP = cv.imread('pedestrians/pos/per%05d.ppm' % (i+1))
    IN = cv.imread('pedestrians/neg/neg%05d.png' % (i+1))
    F = hog(IP)
    HOG_data[i, 0] = 1
    HOG_data[i, 1:] = F
    F = hog(IN)
    HOG_data[i+l_prob, 0] = 0
    HOG_data[i+l_prob, 1:] = F

labels = HOG_data[:, 0]
data = HOG_data[:, 1:]
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(data, labels)
lp = clf.predict(data)

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(0, len(labels)):
    if labels[i] == 1 and lp[i] == 1:
        TP += 1
    elif labels[i] == 0 and lp[i] == 0:
        TN += 1
    elif labels[i] == 0 and lp[i] == 1:
        FP += 1
    else:
        FN += 1

ACC = (TP + TN) / len(labels)
print('Dokładność wynosi:', ACC)

G = cv.imread('testImages/testImage1.png')
G = cv.cvtColor(G, cv.COLOR_BGR2RGB)
img = cv.resize(G, None, fx=0.65, fy=0.65)
img_copy = img.copy()
for i in range(0, int(img.shape[1]-64), 8):
    for j in range(0, int(img.shape[0]-128), 8):
        img_piece = img[j:128+j, i:64+i]
        F = hog(img_piece)
        lp = clf.predict([F])
        if lp == 1:
            cv.rectangle(img_copy, (i, j), (i+64, j+128), (255, 20, 147), 2)

plt.figure()
plt.imshow(img_copy)
plt.axis('off')
plt.savefig('result1.png')
plt.show()