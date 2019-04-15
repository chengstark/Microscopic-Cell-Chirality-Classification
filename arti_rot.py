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


def video_gen():
    imgs = []
    # for filename in os.listdir('arti_rot'):
    #     if filename.endswith(".jpg"):
    #         im = cv2.imread('arti_rot/{}'.format(filename))
    #         print('arti_rot/{}'.format(filename))
    #         imgs.append(im)
    #         print(im.shape)
    im = cv2.imread('arti_rot/0_-20.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/1_-15.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/2_-14.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/3_-13.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/4_-9.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/5_-8.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/6_-7.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/7_-5.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/8_-3.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/9_-2.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/10_-1.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/11_0.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/12_1.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/13_6.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/14_7.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/15_8.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/16_9.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/17_10.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/18_11.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/19_12.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/20_13.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/21_14.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/22_15.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/23_17.jpg')
    imgs.append(im)
    im = cv2.imread('arti_rot/24_19.jpg')
    imgs.append(im)
    img = imgs[0]
    height, width = img.shape[0], img.shape[1]
    video_path = 'arit_rot/a.avi'
    size = (width, height)
    fps = 6
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter("arit_rot/a.avi", fourcc, 6, size)
    i = 0
    for im in imgs:
        video.write(im)
        print('frame {} written!'.format(i))
        i += 1
    video.release()


def make_video():
    imgs = []
    # for i in reversed(range(25)):
    #     print(i)
    #     im = cv2.imread("rev_xy15/frame{}.jpg".format(i))
    #     images.append(im)
    im = cv2.imread('validations/arti-rot/rots/20_0.jpg')
    imgs.append(im)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__() - 1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    nextim = imgs[imgs.__len__()-1]
    row, col, _ = nextim.shape
    rot_mat = cv2.getRotationMatrix2D((row / 2, col / 2), 1, 1.0)
    # result = imutils.rotate(img, angle, (col, row), scale=1.0)
    result = cv2.warpAffine(nextim, rot_mat, (col, row), cv2.INTER_NEAREST)
    imgs.append(result)
    # im = cv2.imread('validations/arti-rot/rots/1_-19.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/2_-18.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/3_-17.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/4_-16.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/5_-15.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/6_-14.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/7_-13.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/8_-12.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/9_-11.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/10_-10.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/11_-9.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/12_-8.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/13_-7.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/14_-6.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/15_-5.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/16_-4.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/17_-3.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/18_-2.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/19_-1.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/20_0.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/21_1.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/22_2.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/23_3.jpg')
    # imgs.append(im)
    # im = cv2.imread('validations/arti-rot/rots/24_4.jpg')
    # imgs.append(im)
    print(imgs.__len__())
    imgs.reverse()
    img = imgs[0]
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 'x264' doesn't work
    video = cv2.VideoWriter("arti_rot/test_rev.avi", fourcc, 6, (width, height))
    for image in imgs:
        video.write(image)

    video.release()


def check():
    a_s = []
    b_s = []
    a = cv2.VideoCapture('arti_rot/a.avi')
    b = cv2.VideoCapture('arti_rot/b.avi')
    while a.isOpened():
        ret, frame = a.read()
        if ret:
            a_s.append(frame)
        else:
            break
        print(1)
    while b.isOpened():
        ret, frame = b.read()
        if ret:
            b_s.append(frame)
        else:
            break

    b_s.reverse()
    a_s = np.asarray(a_s)
    b_s = np.asarray(b_s)
    indicesForMatches = [(i, j) for i, subArrayOfA in enumerate(a_s) for j, subArrayOfB in enumerate(b_s) if
                         np.array_equal(subArrayOfA, subArrayOfB)]
    print(indicesForMatches)

def load_check():
    a = np.load("arti_rot/a_all_frames.npy")
    b = np.load("arti_rot/b_all_frames.npy")
    indicesForMatches = [(i, j) for i, subArrayOfA in enumerate(a) for j, subArrayOfB in enumerate(b) if
                         np.array_equal(subArrayOfA, subArrayOfB)]
    print(indicesForMatches)


def mse(imageA, imageB, degree):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    # print("sum: {} {} {} ".format(np.sum(imageA.astype("float")), np.sum(imageB.astype("float")), degree))
    area = float(imageA.shape[0] * imageA.shape[1])
    err_ = err/area
    # print("{} ERR: {} {} -> {}".format(degree, err, area, err_))
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err_




if __name__ == '__main__':
    # video_gen()
    make_video()
    # check()
    # load_check()
# 5, 1, 1 ,4, 1, 1, 2, 2, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2
