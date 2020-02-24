import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib


def rgb2gray(I):
    return 0.299*I[:,:,0] + 0.587*I[:,:,1] + 0.144*I[:,:,2]

I = plt.imread('mandril.jpg')

#plt.figure(1)
fig, ax = plt.subplots(1)
plt.imshow(I)
plt.title('Mandril')
plt.axis('off')

plt.imsave('mandril.jpg', I)

x = [100, 150, 200, 250]
y = [50, 100, 150, 200]

plt.plot(x, y, 'r.', markersize=10)

rect = Rectangle((50,50),50,100,fill=False, ec='r')
ax.add_patch(rect)
plt.show()

plt.figure(1)
IG = rgb2gray(I)
plt.imshow(IG)
plt.title('Mandril gray')
plt.axis('off')
plt.gray()
plt.show()

plt.figure(1)
_HSV = matplotlib.colors.rgb_to_hsv(I)
plt.imshow(_HSV)
plt.title('Mandril HSV')
plt.axis('off')
plt.show()

