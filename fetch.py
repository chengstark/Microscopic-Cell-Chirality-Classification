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


fetched_lables = []
cnt = 0
f_up = []

def fetch_index(index, index2):
    global cnt
    curr_vec = np.loadtxt("fetch/currs/XY{}_video/vectors.txt".format(index2))
    # for vec in curr_vec:
    #     print(vec)
    #     print("----------------------------------------------------")
    curr_loc = np.loadtxt("fetch/currs/XY{}_video/XY{}_video_loc.txt".format(index2, index2))
    ori_loc = np.loadtxt("new_data/rs/XY{}_video/XY{}_video_loc.txt".format(index2, index2))
    indices = []
    vs = []
    id = 0
    for aloc in curr_loc:
        b_i = 0
        got = False
        for bloc in ori_loc:
            if aloc[0] <= bloc[0] + 5 and aloc[0] >= bloc[0] - 5 and bloc[1] - 5 <= aloc[1] <= bloc[1] + 5:
                indices.append(b_i)
                vs.append(aloc)
                got = True
                break

            b_i += 1
        if not got:
            cnt += 1
            print("fucked up: {} {}".format(aloc[0], aloc[1]))
        print(id)
        id += 1
    # print(indices)
    # print(curr_loc.__len__(), indices.__len__())
    labels = np.loadtxt("new_data/XY{}_index&result.txt".format(index2))
    tmp_label = []
    for i in range(len(curr_vec)):
        for l in labels:
            if l[0] == indices[i]:
                label = l[1]
                tmp_label.append(label)
                f_up.append(curr_vec[i])
                fetched_lables.append(label)

    # print(tmp_label.__len__(), curr_vec.__len__(), labels.__len__())
    # assert curr_loc.__len__() == curr_vec.__len__()
    # assert curr_vec.__len__() == indices.__len__()
    # assert tmp_label.__len__() == curr_vec.__len__()
        # vector = curr_vec[i]
        #
        # print("{}-{}".format(vector, label))

def fectch_ori(i):
    ret = []
    path = "neuro/XY{}_video/XY{}_video_loc.txt".format(i, i)
    loc = np.loadtxt(path)
    label = "neuro/XY{}_index&result.txt".format(i)
    l = np.loadtxt(label)
    for l_ in l:
        tmp = []
        print(l_)
        id, lb = l_
        tmp.append(loc[int(id), 0])
        tmp.append(loc[int(id), 1])
        tmp.append(lb)
        ret.append(tmp)
    np.savetxt("neuro/{}.txt".format(i), ret)
    return ret



if __name__ == '__main__':
    for i in range(1, 33):
        print(i)
        if i < 10:
            # fetch_index(str(i), "0{}".format(i))
            fectch_ori("0{}".format(i))
        else:
            # fetch_index(str(i), str(i))
            fectch_ori(i)


    x = np.loadtxt("neuro/01.txt")
    for x_ in x:
        print(x)
    # np.savetxt("fetch/new_labels.txt", fetched_lables)
    # all_vec = np.loadtxt("fetch/currs/all_vs/all.txt")
    # new_vec = []
    # # for v in all_vec:
    # #     if v not in f_up:
    # #         new_vec.append(v)
    # print("fucked up count: {}".format(cnt))
    # np.savetxt("fetch/new_vecs.txt", f_up)
