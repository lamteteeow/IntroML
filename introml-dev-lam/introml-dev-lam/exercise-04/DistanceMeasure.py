'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    if len(Rx) == len(Ry):
        DR = np.sum(np.abs(Rx - Ry)) / len(Rx)
    return DR


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    if len(Thetax) == len(Thetay):
        sumThetax = np.sum(Thetax)
        sumThetay = np.sum(Thetay)
        l_x = Thetax - sumThetax / len(Thetax)
        l_y = Thetay - sumThetay / len(Thetay)
        l_xx = np.sum(l_x ** 2) / len(Thetax)
        l_yy = np.sum(l_y ** 2) / len(Thetay)
        l_xy = np.sum(l_x * l_y) / len(Thetax)
        DTheta = (1 - (l_xy ** 2 / (l_xx * l_yy))) * 100
    return DTheta
