# Test the trained networks
# Created by yanrpi @ 2018-08-01
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from shutil import copyfile
import os
import filecmp
from os import path
from PIL import Image
# import myresnet
import filecmp
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib import *
from sklearn import manifold, datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import figure
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import kstat
from scipy.stats import tmean
from scipy.stats import sem


def draw_histograms(vec_, label_, i_):
    intensity = vec_
    bins = np.arange(0, 24, 1)
    plt.figure()
    plt.plot(bins, intensity, marker='o', mec='r', mfc='w', label=u'y=x^2曲线图')
    plt.hlines(0, 24, 0.5, colors="c", linestyles="dashed")
    plt.xlim((0, 24))
    plt.ylim((-20, 20))
    plt.xlabel('Frames')
    plt.ylabel('Degree')
    plt.title('Cell {}: {}'.format(i_, label_))
    plt.savefig('dumb_me/{}/Cell_{}.jpg'.format(label_, i_))
    print('{}: histogram written'.format(i_))

def draw_histograms_index(vec1, vec2, vec3, i_):
    intensity = vec1
    bins = np.arange(0, 24, 1)
    plt.figure()
    plt.plot(bins, intensity, marker='s', mec='r', mfc='w', label="CW_SSIM")
    plt.hlines(0, 24, 0.5, colors="c", linestyles="dashed")
    intensity = vec2
    bins = np.arange(0, 24, 1)
    plt.plot(bins, intensity, marker='^', mec='g', mfc='w', label="SSIM")
    plt.hlines(0, 24, 0.5, colors="c", linestyles="dashed")
    intensity = vec3
    bins = np.arange(0, 24, 1)
    plt.plot(bins, intensity, marker='o', mec='b', mfc='w', label="MSE")
    plt.hlines(0, 24, 0.5, colors="c", linestyles="dashed")
    plt.legend(loc='lower right')
    plt.xlim((0, 24))
    plt.ylim((-20, 20))
    plt.xlabel('Frames')
    plt.ylabel('Degree')
    plt.title('[Cell {}]'.format(i_))
    plt.savefig('index_compare/Cell_{}.jpg'.format( i_))
    print('{}: histogram written'.format(i_))

if __name__ == '__main__':
    # vectors = np.loadtxt("fetch/new_vecs.txt")
    # labels = np.loadtxt("fetch/new_labels.txt")
    # i = 0
    # for vec in vectors:
    #     label = int(labels[i])
    #     if label == 0:
    #         # CW
    #         label = str(label) + "_CW"
    #     if label == 1:
    #         # CCW
    #         label = str(label) + "_CCW"
    #     if label == 2:
    #         # OTH & CPLX
    #         label = str(label) + "_OTH_CPLX"
    #     draw_histograms(vec, label, i)
    #     i += 1
    cw = np.loadtxt("index_compare/cws.txt")
    ssim = np.loadtxt("index_compare/ssims.txt")
    mse = np.loadtxt("index_compare/mses.txt")

    for i in range(0, cw.shape[0]):
        draw_histograms_index(cw[i], ssim[i], mse[i], i+1)





