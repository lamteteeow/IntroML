import numpy as np
#
# NO OTHER IMPORTS ALLOWED
#


def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    hist, bin_edges = np.histogram(img, bins=np.arange(257), density=False)
    return hist.T


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    binar = np.where(img > t, 255, 0)
    return binar


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    p0 = np.sum(hist[:theta + 1])
    p1 = np.sum(hist[theta + 1:])
    return p0, p1


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    edges = np.arange(len(hist))
    if p0 != 0:
        mu0 = np.sum(edges[:theta + 1] * hist[:theta + 1]) / p0
    else:
        mu0 = 0
    if p1 != 0:
        mu1 = np.sum(edges[theta + 1:] * hist[theta + 1:]) / p1
    else:
        mu1 = 0
    return mu0, mu1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables
    # all_inter_var = np.empty(shape=0)
    inter_var = 0
    otsu_theta = 0
    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1
    binar_hist = hist / np.sum(hist)

    # TODO loop through all possible thetas
    for theta in range(len(hist)):
        # TODO compute p0 and p1 using the helper function
        p0, p1 = p_helper(binar_hist, theta)
        # TODO compute mu and m1 using the helper function
        mu0, mu1 = mu_helper(binar_hist, theta, p0, p1)
        # TODO compute variance
        # np.append(all_inter_var, (p0 * p1 * (mu1 - mu0)**2))

        # TODO update the threshold
        if (p0 * p1 * (mu1 - mu0)**2) > inter_var:
            inter_var = p0 * p1 * (mu1 - mu0) ** 2
            otsu_theta = theta
    return otsu_theta


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    # TODO
    return binarize_threshold(img, calculate_otsu_threshold(create_greyscale_histogram(img)))
