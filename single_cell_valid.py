import urllib.request
import cv2
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

if __name__ == '__main__':
    print("original: ")
    n = np.loadtxt('arti_rot/vectors.txt')
    a = n[0]
    b = n[1]
    print(a)
    print(b)
    print("reversed: ")
    b = np.flip(b, axis=0)
    b = np.multiply(b, -1)
    a = np.multiply(a, -1)
    print(a)
    print(b)
