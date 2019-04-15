# import urllib.request
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import time
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
import sklearn
import cv2
import copy
import numpy as np
import os
import copy
import math
import time
# import pandas as pd
# from joblib import dump, load
from sklearn.model_selection import train_test_split
# from os import path
# from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
# import math
# from sklearn.cluster import KMeans
from sklearn import svm
# from sklearn.svm import SVC
# from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from numpy import genfromtxt
# import simple_cnn as sic
# import imutils


def get_data(img, i):
    adata_ = []
    im = img.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 55, 255, cv2.THRESH_BINARY)
    cv2.imwrite("my/my2/{}.jpg".format(i), im)
    for arow_ in im:
        arow = arow_.tolist()
        print(arow)
        if 255 in arow:
            initial = arow.index(255)
            cpy_row = copy.deepcopy(arow)
            final = len(cpy_row) - 1 - cpy_row[::-1].index(255)
            dis = final - initial
            adata_.append(dis)
        else:
            adata_.append(0)
    print(adata_)
    return adata_


if __name__ == '__main__':
    video = cv2.VideoCapture('cell1_nor.avi')
    frames = []
    col = None
    row = None
    data = []
    i = 0
    while True:
        (grabbed, frame) = video.read()
        if not grabbed:
            break

        frames.append(frame)
        col, row, _ = frame.shape
        f = frame.copy()
        f_ = f[int(5):int(col - 5), int(5):int(row - 5)]
        cv2.imwrite("my/{}.jpg".format(i), f_)
        adata = get_data(f_, i)
        data.append(adata)
        i += 1

    plt_i = 0
    for d in data:
        max_ = max(d)
        max_i = d.index(max_)
        plt.axvline(x=max_i)
        plt.plot(range(col - 10), d)
        plt.savefig("my/plt/{}.jpg".format(plt_i))
        plt.clf()
        plt_i += 1

