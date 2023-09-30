# Implement the histogram equalization in this file
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('hello.png', cv2.IMREAD_GRAYSCALE)
hist = np.zeros(shape=256)
for y in range(len(img)):
    for x in range(len(img[0])):
        hist[img[y, x] - 1] += 1 #is [y, x] better than [y][x] ?
# print(img.shape)
# print(img)
# print(img[631, 688])
# print(hist)
# print(hist[142])

hist01 = hist / np.sum(hist)

cum_dis = np.zeros(shape=256)
for i in range(256):
    cum_dis[i] = np.sum(hist01[0:i])
# print(cum_dis)
C_min_pos = np.argmin(np.where(cum_dis == 0, 2, cum_dis)) + 1  #shouldn't be worry about Grenzbedingung
# print(C_min_pos)
C_min = cum_dis[C_min_pos]
# print(C_min)
# print(np.argmax(cum_dis))
new_cat = np.zeros(shape=img.shape)
for y in range(len(new_cat)):
    for x in range(len(new_cat[0])):
        new_cat[y, x] = np.abs((cum_dis[img[y, x]] - C_min) * 255 / (1 - C_min))
# print(new_cat)

# plt.plot(hist)
# plt.plot(np.where(cum_dis == 0, 2, cum_dis))
# plt.plot(cum_dis)

# plt.subplot(1, 2, 1)
# plt.imshow(img, 'gray')
# plt.title('Original')
# if new_cat is not None:
#     plt.subplot(1, 2, 2)
#     plt.imshow(new_cat, 'gray')
#     plt.title('Histogram equalized')
# plt.show()

cv2.imwrite('kitty.png', new_cat)
