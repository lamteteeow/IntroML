'''
Created on 25.11.2017
Modified on 05.12.2020

@author: Daniel, Max, Charly, Mathias
'''
import cv2
import matplotlib.pyplot as plt
from otsu import otsu


img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)

# print(img.shape)
# histt, bins = np.histogram(img, bins=256, density=False)
# hist = histt.T
# boo = np.abs(hist[160] - 8416)
# print(boo)
# print(hist[:17])
# plt.plot(hist)
# plt.show()

# histogram = otsu.create_greyscale_histogram(img)
# print(histogram)
# a = 0
# for i in range(len(histogram)):
#     a = a + histogram[i]
# print(len(histogram))
# print(a)

# a1 = np.arange(10)
# mu0, mu1 = otsu.mu_helper(a1, 4, 0.4, 0.6)
# print(mu0, mu1)

# a = np.array([5, 1, 0, 13])
# b = np.array([7, 2, 14, 2])
# print(a*b)

# print(np.zeros((1, 1)))
# print(np.append(np.array([]), 1))
# print(np.empty(shape=0))

res = otsu(img)

plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.title('Original')
if res is not None:
    plt.subplot(1, 2, 2)
    plt.imshow(res, 'gray')
    plt.title('Otsu\'s - Threshold = 120')
plt.show()
