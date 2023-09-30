import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    # TODO
    # write the kernel
    # kernel = np.zeros(shape=(ksize, ksize), dtype=float)
    # half_k = ksize // 2
    # for i in range(-half_k, half_k + 1):
    #     for j in range(-half_k, half_k + 1):
    #         kernel[i, j] = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    #
    # # normalize the kernel
    # kernel = kernel / np.sum(np.sum(kernel))
    #
    # # convolution between kernel and image
    # img_out = convolve(img_in, kernel).astype(int)
    # return kernel, img_out
    kernel = np.zeros((ksize, ksize))
    sum = 0
    s = 2 * (sigma ** 2)
    half_size = int(ksize / 2)
    # calculating Filter Kernel after formula
    for i in range(-half_size, half_size + 1):
        for j in range(-half_size, half_size + 1):
            r = np.sqrt(i * i + j * j)
            kernel[i + half_size][j + half_size] = (np.exp(-(r * r) / s)) / (np.pi * s)
            sum += kernel[i + half_size][j + half_size]
    # normalization
    kernel = kernel / sum
    blurred_image = convolve(img_in, kernel, mode='constant', cval=0.0)
    return kernel, blurred_image.astype(int)


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # TODO
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gx = convolve(img_in, g_x)
    gy = convolve(img_in, g_y)
    return gx.astype(int), gy.astype(int)


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # TODO
    # gradient = np.sqrt(gx ** 2 + gy ** 2)
    # theta = np.arctan2(gy, gx)
    # return gradient.astype(int), theta
    shape = np.shape(gx)
    gradient_magnitude = np.zeros((shape[0], shape[1]))
    theta = np.zeros((shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            gradient_magnitude[i][j] = np.sqrt(((gx[i][j]) ** 2) + ((gy[i][j]) ** 2))
            theta[i][j] = np.arctan2(gy[i][j], gx[i][j])
    return gradient_magnitude.astype(int), theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # TODO
    degree = angle * (180 / np.pi)
    degree %= 180
    if degree < 22.5 or degree >= 157.5:
        degree = 0
    elif 22.5 <= degree < 67.5:
        degree = 45
    elif 67.5 <= degree < 112.5:
        degree = 90
    else:
        degree = 135
    return degree


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    max_sup = np.zeros_like(g)
    final = np.zeros_like(theta)
    for i in range(len(theta)):
        for j in range(len(theta[0])):
            final[i][j] = convertAngle(theta[i][j])

    for i in range(1, len(g) - 1):
        for j in range(1, len(g[0]) - 1):
            if final[i, j] == 0:
                if g[i, j] >= g[i, j - 1] and g[i, j] >= g[i, j + 1]:
                    max_sup[i, j] = g[i, j]
            elif final[i, j] == 45:
                if g[i, j] >= g[i - 1, j + 1] and g[i, j] >= g[i + 1, j - 1]:
                    max_sup[i, j] = g[i, j]
            elif final[i, j] == 90:
                if g[i, j] >= g[i - 1, j] and g[i, j] >= g[i + 1, j]:
                    max_sup[i, j] = g[i, j]
            elif final[i, j] == 135:
                if g[i, j] >= g[i + 1, j + 1] and g[i, j] >= g[i - 1, j - 1]:
                    max_sup[i, j] = g[i, j]
    return max_sup


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    # TODO
    hysteris = np.zeros_like(max_sup)
    threshimg = np.zeros_like(max_sup)
    for i in range(len(max_sup)):
        for j in range(len(max_sup[0])):
            if max_sup[i, j] <= t_low:
                threshimg[i, j] = 0
            elif max_sup[i, j] > t_high:
                threshimg[i, j] = 2
            else:
                threshimg[i, j] = 1
    #
    # for i in range(len(max_sup)):
    #     for j in range(len(max_sup[0])):
    #         if threshimg[i, j] == 2:
    #             hysteris[i, j] = 255
    #             if len(max_sup) > i > 0 and len(max_sup[0]) > j > 0:
    #                 if max_sup[i - 1, j - 1] > t_low:
    #                     hysteris[i - 1, j - 1] = 255
    #                 if max_sup[i - 1, j] > t_low:
    #                     hysteris[i - 1, j] = 255
    #                 if max_sup[i - 1, j + 1] > t_low:
    #                     hysteris[i - 1, j + 1] = 255
    #                 if max_sup[i, j - 1] > t_low:
    #                     hysteris[i, j - 1] = 255
    #                 if max_sup[i, j + 1] > t_low:
    #                     hysteris[i, j + 1] = 255
    #                 if max_sup[i + 1, j - 1] > t_low:
    #                     hysteris[i + 1, j - 1] = 255
    #                 if max_sup[i + 1, j] > t_low:
    #                     hysteris[i + 1, j] = 255
    #                 if max_sup[i + 1, j + 1] > t_low:
    #                     hysteris[i + 1, j + 1] = 255
    # return hysteris


    result = np.zeros(threshimg.shape)
    # pad thresimg image with zeros
    padded_array = np.pad(threshimg, (1, 1), constant_values=(0, 0))
    for a in range(1, padded_array.shape[0] - 1):
        for b in range(1, padded_array.shape[1] - 1):
            if padded_array[a][b] == 2:
                # attention - different moving indexing
                result[a-1][b-1] = 255
                if padded_array[a][b-1] == 1:
                    result[a - 1][b-1 - 1] = 255
                if padded_array[a][b+1] == 1:
                    result[a - 1][b+1 - 1] = 255
                if padded_array[a+1][b-1] == 1:
                    result[a+1 - 1][b-1 - 1] = 255
                if padded_array[a-1][b+1] == 1:
                    result[a-1 - 1][b+1 - 1] = 255
                if padded_array[a+1][b] == 1:
                    result[a+1 - 1][b - 1] = 255
                if padded_array[a-1][b] == 1:
                    result[a-1 - 1][b - 1] = 255
                if padded_array[a-1][b-1] == 1:
                    result[a-1 - 1][b-1 - 1] = 255
                if padded_array[a+1][b+1] == 1:
                    result[a+1 - 1][b+1 - 1] = 255
    #print("Padded array:")
    #print(padded_array)
    #print("result")
    #print(result)

    return result



def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result


if __name__ == '__main__':
    img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
    res = canny(img)

