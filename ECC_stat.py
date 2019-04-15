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
from scipy import stats
import sys


def draw_histograms(deg1, deg2, cell_index):
    intensity = deg1
    intensity2 = deg2
    bins = np.arange(0, 24, 1)
    plt.figure()
    plt.plot(bins, intensity, marker='o', mec='r', mfc='w', label=u'y=x^2曲线图')
    plt.hlines(0, 24, 0.5, colors="c", linestyles="dashed")
    plt.plot(bins, intensity2, marker='^', mec='r', mfc='w', label=u'y=x^2曲线图')
    plt.hlines(0, 24, 0.5, colors="k", linestyles="dashed")
    plt.xlim((0, 24))
    plt.ylim((-20, 20))
    plt.xlabel('Frames')
    plt.ylabel('Degree')
    plt.title('Cell {}: Rotation Degrees'.format(cell_index))
    plt.savefig('ECC/TREND/Cell{}.jpg'.format(cell_index))


def ECC_stat():
    ecc_vec = np.loadtxt("ECC/ECC/vectors.txt")
    ori_vec = np.loadtxt("ECC/cmp/vectors.txt")
    N = ecc_vec.__len__()
    var_a = ecc_vec.var(ddof=1)
    var_b = ori_vec.var(ddof=1)
    s = np.sqrt((var_a + var_b) / 2)
    t = (ecc_vec.mean() - ori_vec.mean()) / (s * np.sqrt(2 / ecc_vec.__len__()))
    df = 2 * N - 2

    # p-value after comparison with the t
    p = 1 - stats.t.cdf(t, df=df)

    print("t = " + str(t))
    print("p = " + str(2 * p))
    ### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.

    ## Cross Checking with the internal scipy function
    t2, p2 = stats.ttest_ind(ecc_vec, ori_vec)
    t_mean = sum(t2)/t2.__len__()
    p_mean = sum(p2)/p2.__len__()
    print("t = " + str(t2))
    print("p = " + str(p2))
    print("t_mean = " + str(t_mean))
    print("p_mean = " + str(p_mean))

    for i in range(ecc_vec.__len__()):
        draw_histograms(ecc_vec[i], ori_vec[i], i)

if __name__ == '__main__':
    ECC_stat()
    # draw_histograms()

