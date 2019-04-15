# import urllib.request
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import time
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
import sklearn
import statistics
import cv2
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



frames = []


def get_frames():
    frame_idx = 0
    cap = cv2.VideoCapture("cell1_nor.avi")
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imwrite("sift_test/frame{}.jpg".format(frame_idx), frame)
        frames.append(frame)
        frame_idx += 1


def sift():
    for i in range(0, len(frames) - 1):
        im = frames[i]
        im2 = frames[i+1]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

        sift = cv2.xfea




