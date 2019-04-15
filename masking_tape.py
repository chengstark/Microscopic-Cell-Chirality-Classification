import urllib.request
import cv2
import imutils
from pylab import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
from collections import Counter
import time
from os import path
from os import walk
import pickle
import sklearn
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import copy
from sklearn.svm import SVC
# import simple_cnn as sic
# import imutils

import sys
sys.setrecursionlimit(100)
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication, QMessageBox
from PyQt5.QtWidgets import QMenu, QAction, QMainWindow, QLabel, QFileDialog
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit
from PyQt5.QtGui import QPixmap, QFont
import statistics
# from PyQt5 import QtWidgets, QtGui
import time
import cv2
import shutil


def recenter(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=50, param2=15,
                               minRadius=10, maxRadius=40)
    cimg = img.copy()
    for i in circles[0, :]:
        ''' Draw boundary circles '''
        cv2.circle(cimg, (i[0], i[1]), i[2], (255, 0, 0), 3)
        ''' Draw the centers of the circles '''
        cv2.circle(cimg, (i[0], i[1]), 2, (255, 0, 0), 5)
    cv2.imwrite('recenter/test.jpg', cimg)
    print(circles[0, :][0][0], circles[0, :][0][1])
    return circles[0, :][0][0], circles[0, :][0][1]


def masking_tape(imga_ori):
    imga = cv2.cvtColor(imga_ori, cv2.COLOR_BGR2GRAY)
    _, imga_t = cv2.threshold(imga, 55, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('maskinh_tape_strategy/imga_threshed.jpg', imga_t)
    _, cnts, h = cv2.findContours(imga_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(h)
    # print(cnts)
    # cv2.fillPoly(imga_ori, cnts, (0, 0, 255))
    mask = np.zeros_like(imga)
    cv2.drawContours(mask, cnts, -1, 204, cv2.FILLED)
    out = np.zeros_like(imga)
    out[mask == 204] = imga[mask == 204]

    # cv2.imwrite('maskinh_tape_strategy/cnted_imga_threshed.jpg', out)
    return out


def mse(imageA, imageB, degree):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    print("sum: {} {} {} ".format(np.sum(imageA.astype("float")), np.sum(imageB.astype("float")), degree))
    area = float(imageA.shape[0] * imageA.shape[1])
    err_ = err/area
    # print("{} ERR: {} {} -> {}".format(degree, err, area, err_))
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err_


if __name__ == '__main__':
    imga_ori = cv2.imread("validations/arti-rot/rots/21_1.jpg")
    imgb_ori = cv2.imread("validations/arti-rot/rots/22_2.jpg")
    recenter(imga_ori)
    im_a_cvted = masking_tape(imga_ori)
    im_b_cvted = masking_tape(imgb_ori)

    print(mse(im_a_cvted, im_b_cvted, 1))




