import numpy as np
import cv2
from math import floor

# Main function for running file as main
def compute_histogram(img):
    '''Computes the histogram for a given input grayscale image
    1. Create buffer array (Integer type)
    2. extract image dimensions'''
    hist = np.zeros(256, dtype=int)
    height, width = img.shape
    for k in range(height):
        for l in range(width):
            idx = img[k][l]
            hist[idx] += 1;
    return hist

def compute_cum_dist(img, hist):
    '''Compute cumulative distribution for a given image and histogram
    1. Copy Histogramm and divide by image dimensions to get an array of probabilities
    2. Iterate over the new array to add up the probabilities'''
    cumDistr = hist.copy()/(img.shape[0]*img.shape[1])
    for i in range(1, len(hist)):
        cumDistr[i] += cumDistr[i-1]
    return cumDistr

def hist_equalization(img, cumDistr):
    '''Compute the histogram equalization
    1. Extract minimum (nonzero) cumulative distribution
    2. Calculate divident (to save computing power later)
    3. iterate over pixels and calculate the new grayscale values using the formula'''
    cMin = np.amin(cumDistr[cumDistr > 0])
    divid = 1-cMin
    height, width = img.shape
    for k in range(height):
        for l in range(width):
            idx = img[k][l]
            img[k][l] = floor((cumDistr[idx] - cMin) / divid * 255)
    return img

def main():
    '''1. Read in the image using opencv, the second parameter is there to remove color
    channels, if existend.
    2. Compute histogramm bins (number of occurances)'''
    img = cv2.imread('hello.png', cv2.IMREAD_GRAYSCALE)
    hist = compute_histogram(img)
    cumDistr = compute_cum_dist(img, hist)
    resImg = hist_equalization(img, cumDistr)
    cv2.imwrite('kitty.png', resImg)
            
if __name__ == '__main__':
    main()
