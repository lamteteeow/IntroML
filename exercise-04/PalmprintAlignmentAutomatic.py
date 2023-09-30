'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


# do not import more modules!


def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cir = cv2.circle(img, (x, y), 5, 255, 2)
    # plt.imshow(cir)
    # plt.show
    return cir


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    ret, binmap = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)
    return cv2.GaussianBlur(binmap, (5, 5), 0, cv2.BORDER_DEFAULT)


def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))

    # find the largest contour
    c = max(contours, key=cv2.contourArea)

    # create an image for the contour
    contour = np.zeros(img.shape)

    cv2.drawContours(contour, [c], 0, 255, 2)
    # plt.imshow(contour)
    # plt.show()
    return contour


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    # create 1D numpy array for storing y_values
    y_values = np.zeros(255)
    # the counters have more than 1 black pixel - I just want to count the first black pixel
    helper = 255
    # count of intersecting y-values
    count = 0
    # we just need 6 y-values
    y_size = contour_img.shape[0]
    for y in range(y_size):
        if contour_img[y][x] != 255:
            helper = 255
        if contour_img[y][x] == 255 and helper == 255:
            # must change helper to 0 otherwise I count second, third black pixel in the same counter
            helper = 0
            y_values[count] = y
            count = count + 1
    # print(y_values)
    # choose just the 6 intersecting y-values starting from 1 because the lowest intersection should NOT count
    y_values = y_values[1:7]
    # print(y_values)
    return y_values


def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''
    # measure the angle it makes with a horizontal line
    direction_vector = (y2 - y1) / (x2 - x1)

    # I do not understand ??
    difference = y1 - (direction_vector * x1)
    # x_value increase by 1
    x = 0

    # starting point
    # I do not understand ??
    # y = (direction_vector * x) + difference

    while True:
        # I do not understand ??
        y = int((direction_vector * x) + difference)
        if img[y, x] == 255:
            break
        x = x + 1
    return (y, x)


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    # x_value increase by 1
    x = 0

    # from k3 to k1
    # measure the angle it makes with a horizontal line
    direction_vector = (k3[0] - k1[0]) / (k3[1] - k1[1])
    difference = k3[0] - (direction_vector * k3[1])
    # starting point
    # y1 = (direction_vector * x) + difference
    # y1 = int(y1)

    # from k3 to k1 via k2
    # from vertical to horizontal
    direction_vector_k2 = -1 / direction_vector
    difference_k2 = k2[0] - (direction_vector_k2 * k2[1])
    # starting point
    # y2 = (direction_vector_k2 * x) + difference_k2
    # y2 = int(y2)

    # I do not understand ??
    factor = np.inf

    while True:
        y1 = (direction_vector * x) + difference
        y1 = int(y1)
        y2 = (direction_vector_k2 * x) + difference_k2
        y2 = int(y2)

        # I do not understand ??
        difference_y = y1 - y2
        if factor < np.abs(difference_y):
            break
        # moving from 0 column to the right step 1
        x = x + 1
        factor = np.abs(y1 - y2)

    # calculate angle
    # deltaY = k3[0] - k1[0]
    # deltaX = k3[1] - k1[1]
    # radians = math.atan2(deltaY, deltaX)
    # angle = radians * (180/math.pi)
    # Set scale to 1 when computing the transform ????????

    center = (y1, x)
    # calculate radians
    radians = np.arctan(direction_vector_k2)
    # calculate degree
    angle = np.degrees(radians)

    rotated_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

    return rotated_matrix


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    # TODO threshold and blur

    # TODO find and draw largest contour in image

    # TODO choose two suitable columns and find 6 intersections with the finger's contour

    # TODO compute middle points from these contour intersections

    # TODO extrapolate line to find k1-3

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3

    # TODO rotate the image around new origin
    # threshold and blur
    preprocessed_image = binarizeAndSmooth(img)

    # find and draw largest contour in image
    largest_contour = drawLargestContour(preprocessed_image)

    # choose two suitable columns and find 6 intersections with the finger's contour
    # 6 and 7 did not work for the hand 3
    x1 = 11
    x2 = 12
    y_values_1 = getFingerContourIntersections(largest_contour, x1)
    y_values_2 = getFingerContourIntersections(largest_contour, x2)

    # compute middle points from these contour intersections
    middle_points_1 = np.zeros(3)
    middle_points_2 = np.zeros(3)
    # 6 values and we calculate only a pair hence +2
    a = 0
    for i in range(3):
        middle_points_1[i] = y_values_1[a] + np.abs(y_values_1[a] - y_values_1[a + 1]) / 2
        a = a + 2
    b = 0
    for j in range(3):
        middle_points_2[j] = y_values_2[b] + np.abs(y_values_2[b] - y_values_2[b + 1]) / 2
        b = b + 2

        # extrapolate line to find k1-3
    k1 = findKPoints(largest_contour, middle_points_1[0], x1, middle_points_2[0], x2)
    k2 = findKPoints(largest_contour, middle_points_1[1], x1, middle_points_2[1], x2)
    k3 = findKPoints(largest_contour, middle_points_1[2], x1, middle_points_2[2], x2)

    # calculate Rotation matrix from coordinate system spanned by k1-3
    rotated_matrix = getCoordinateTransform(k1, k2, k3)

    # rotate the image around new origin
    # Rotated palm must have the same dimensions as input
    final_image = cv2.warpAffine(img, rotated_matrix, (img.shape[1], img.shape[0]))
    # plt.imshow(final_image)
    # plt.show()

    return final_image


# if __name__ == '__main__':
#     img1 = cv2.imread('Hand1.jpg', cv2.IMREAD_GRAYSCALE)
#     img2 = cv2.imread('Hand2.jpg', cv2.IMREAD_GRAYSCALE)
#     img3 = cv2.imread('Hand3.jpg', cv2.IMREAD_GRAYSCALE)
#     bin_img1 = binarizeAndSmooth(img1)
#     plt.imshow(img1, 'gray')
#     plt.show()
#     plt.imshow(bin_img1, 'gray')
#     plt.show()
#     # plt.imshow(drawCircle(bin_img1, int(bin_img1.shape[0]/2), int(bin_img1.shape[1]/2)), 'gray')
#     # plt.show()
#     # plt.imshow(cv2.circle(img1, (int(bin_img1.shape[0] / 2), int(bin_img1.shape[1] / 2)), 5, 255, 2), 'gray')
#     # plt.show()
#     plt.imshow(drawLargestContour(bin_img1), 'gray')
#     plt.show()

