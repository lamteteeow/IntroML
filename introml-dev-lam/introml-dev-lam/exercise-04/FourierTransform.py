'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt


# do not import more modules!


def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    x = r * np.cos(theta) + shape[1] / 2
    y = r * np.sin(theta) + shape[0] / 2
    return y, x


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    :param img:
    :return:
    '''
    # print(img)
    # print(fou)
    # print(np.abs(fou))
    # fshift = np.fft.fftshift((np.abs(fou)))
    # print(fshift)
    # ifou = np.fft.ifft2(np.abs(fou))
    # print(ifou)
    fool = np.fft.fftshift(np.abs(np.fft.fft2(img)))
    fooled = np.where(fool == 0, 10**-10, fool)
    return 20 * np.log10(fooled)


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''
    R = np.zeros(k, float)
    # for i in range(1, k + 1):
    #     theta = 0
    #     while theta <= np.pi:
    #         for r in range(k * (i - 1), k * i + 1):
    #             coordinate = polarToKart(magnitude_spectrum.shape, r, theta)
    #             R[i - 1] += magnitude_spectrum[int(coordinate[0]), int(coordinate[1])]
    #         theta += np.pi / (sampling_steps - 1)
    for i in range(1, k + 1):
        for step in range(sampling_steps):
            theta = np.pi * step / (sampling_steps - 1)
            for r in range(k * (i - 1), k * i + 1):
                coordinate = polarToKart(magnitude_spectrum.shape, r, theta)
                R[i - 1] += magnitude_spectrum[int(coordinate[0]), int(coordinate[1])]
    return R


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area
    :return: feature vector of length k
    """
    Theta = np.zeros(k, float)
    for i in range(1, k + 1):
        for step in range(sampling_steps):
            theta = i * step / (sampling_steps - 1)
            for r in range(int(np.min(magnitude_spectrum.shape) / 2)):
                coordinate = polarToKart(magnitude_spectrum.shape, r, theta * np.pi / k)
                Theta[i - 1] += magnitude_spectrum[int(coordinate[0]), int(coordinate[1])]
    return Theta


def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    magspec = calculateMagnitudeSpectrum(img)
    R = extractRingFeatures(magspec, k, sampling_steps)
    Theta = extractFanFeatures(magspec, k, sampling_steps)
    return R, Theta


# if __name__ == '__main__':
#     #     img = np.zeros((10, 10))
#     #     # img[5] = 255
#     #     # img[:, 5] = 255
#     #     img[4:6, 4:6] = 1
#     #     spectrum = calculateMagnitudeSpectrum(np.copy(img))
#     fan = np.zeros((90, 100))
#     fan[:, :50] = 255
#     print(extractFanFeatures(fan, 4, 10))
