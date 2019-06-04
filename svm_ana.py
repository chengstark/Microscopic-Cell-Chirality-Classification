import urllib.request
import cv2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
from sklearn.decomposition import PCA
from collections import Counter
import time
from os import path
from os import walk
import pickle
import sklearn
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import copy
from sklearn.svm import SVC
# import simple_cnn as sic
# import imutils

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import statistics
import sys
import random
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import imageio

total_cw = 0
total_ccw = 0
total_oth = 0

pred_total_cw = 0
pred_total_ccw = 0
pred_total_oth = 0

sample100_id = 0

train_set_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
test_set_index = [15, 16, 17, 18, 32]


def svm_classifier(vec, lab, name, class_weight_=None):
    print(vec.shape)
    print(lab.shape)
    gt_vectors = vec
    gt_labels = lab

    # cross_val_lab = np.reshape(lab, (-1, 1))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(gt_vectors, gt_labels, test_size=0.15, random_state=50)
    if class_weight_ is None:
        my_svm = sklearn.svm.SVC(kernel='poly', random_state=0, tol=1e-50, max_iter=500000, C=1, probability=True)
        cross_val_svm = sklearn.svm.LinearSVC(random_state=10, tol=1e-20, max_iter=500000, C=1)

    else:
        my_svm = sklearn.svm.LinearSVC(random_state=10, tol=1e-20, max_iter=500000, C=1, class_weight=class_weight_)
        cross_val_svm = sklearn.svm.LinearSVC(random_state=10, tol=1e-20, max_iter=500000, C=1, class_weight=class_weight_)

    print(X_train.shape, y_train.shape)
    my_svm.fit(X_train, y_train)
    with open('ideal_train/{}.pickle'.format(name), 'wb') as f:
        pickle.dump(my_svm, f)
        print('{} SVM model saved!'.format(name))
    with open('ideal_train/{}.pickle'.format(name), 'rb') as f:
        my_svm = pickle.load(f)
    train_result = my_svm.predict(X_train)
    test_result = my_svm.predict(X_test)
    train_acc = (y_train == train_result).sum() / float(y_train.size)
    test_acc = (y_test == test_result).sum() / float(y_test.size)
    print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    return train_acc, test_acc, my_svm


def data_process(video_index):
    # test = np.loadtxt('gt_labels.txt')
    # labels_ipt = np.loadtxt('new_data/XY{}_index&result.txt'.format(video_index))
    # vectors_ipt = np.loadtxt('new_data/rs/XY{}_video/vectors.txt'.format(video_index))
    # print(labels_ipt[-1][0] + 1)
    # size = int(labels_ipt[-1][0]) + 1
    # labels = np.full((size,), 99999.0)
    # vectors = np.full((size, 24), 99999.0)
    # for content in labels_ipt:
    #     index = content[0]
    #     index = int(index)
    #     alabel = content[1]
    #     print(index)
    #     labels[index] = alabel
    #     vectors[index] = vectors_ipt[index]
    # np.savetxt('new_processed/nv/{}.txt'.format(video_index), vectors)
    # np.savetxt('new_processed/nl/{}.txt'.format(video_index), labels)
    labels = np.loadtxt('new_processed/nl/catl.txt')
    l_rev = labels.copy()
    np.fliplr(l_rev)
    fl = np.concatenate((labels,l_rev), axis=0)
    print(labels.shape)
    vectors = np.loadtxt('new_processed/nv/catv.txt')
    v_rev = vectors.copy()
    np.fliplr(v_rev)
    fv = np.concatenate((vectors,v_rev),axis=0)
    index = 0
    for x in labels:
        if x == 99999.0:
            print(x)
            labels = np.delete(labels, (index), axis=0)
        else:
            index+=1

    i = 0
    for x in vectors:
        if x[0] == 99999.0:
            print(x)
            vectors = np.delete(vectors, (i), axis=0)
        else:
            i += 1
    print(i)
    print(vectors.shape)
    np.savetxt('new_processed/finalv.txt', vectors)
    np.savetxt('new_processed/finall.txt', labels)


def test_video(idx, rot_non_name, O1_name, rot_non_name_count, O1_name_count):
    # vec = np.loadtxt("/Users/cheng_stark/tmp/Results_with_SEG_05062019/XY{}_reversed/vectors.txt".format(idx))
    vec = np.loadtxt("/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt".format(idx))
    vec_count = []
    for v in vec:
        p_count = 0
        n_count = 0
        z_count = 0
        for x in v:
            if x > 0:
                p_count += 1
            elif x < 0:
                n_count += 1
            elif x == 0:
                z_count += 1
        vec_count.append([p_count, n_count, z_count])

    vec_count = np.asarray(vec_count)

    with open('exp_svm/{}'.format(rot_non_name), 'rb') as f:
        rot_non_model = pickle.load(f)

    with open('exp_svm/{}'.format(O1_name), 'rb') as f:
        O1_model = pickle.load(f)

    with open('exp_svm/{}'.format(rot_non_name_count), 'rb') as f:
        rot_non_model_count = pickle.load(f)

    with open('exp_svm/{}'.format(O1_name_count), 'rb') as f:
        O1_model_count = pickle.load(f)

    rot_non_prediction = rot_non_model.predict(vec)
    rot_non_prediction_count = rot_non_model_count.predict(vec_count)

    print(rot_non_prediction_count)

    # 0 ROT || 1 NON
    idx_rot = []
    idx_rot_count = []
    idx_non = []
    idx_non_count = []
    for i in range(0, rot_non_prediction.shape[0]):
        if rot_non_prediction[i] == 0:
            idx_rot.append(i)
        elif rot_non_prediction[i] == 1:
            idx_non.append(i)

        if rot_non_prediction_count[i] == 0:
            idx_rot_count.append(i)
        elif rot_non_prediction_count[i] == 1:
            idx_non_count.append(i)

    print(idx_rot)
    print(idx_non)

    test_rot = []
    test_rot_count = []
    for id in idx_rot:
        test_rot.append(vec[id])
    for id in idx_rot_count:
        test_rot_count.append(vec_count[id])
    test_rot = np.asarray(test_rot)
    test_rot_count = np.asarray(test_rot_count)

    rot_pred = O1_model.predict(test_rot)
    rot_pred_count = O1_model_count.predict(test_rot_count)

    result = np.zeros(vec.shape[0])
    result_count = np.zeros(vec.shape[0])

    for i in range(0, len(rot_pred)):
        id = idx_rot[i]
        result[id] = rot_pred[i]
    for i in range(0, len(rot_pred_count)):
        id = idx_rot_count[i]
        result_count[id] = rot_pred_count[i]

    for i in idx_non:
        result[i] = 2
    for i in idx_non_count:
        result_count[i] = 2

    # print(result)
    # print(result_count)

    expected = np.loadtxt("const_index/lab/{}.txt".format(idx))
    expected = expected[:, 2]
    print(result.shape)
    print(result_count.shape)
    print(expected.shape)

    err = 0
    cw_err = 0
    ccw_err = 0
    oth_err = 0

    err_count = 0
    cw_err_count = 0
    ccw_err_count = 0
    oth_err_count = 0

    bias_cwasccw = 0
    bias_cwasoth = 0
    bias_ccwascw = 0
    bias_ccwasoth = 0
    bias_othascw = 0
    bias_othasccw = 0
    bias_cwascw = 0
    bias_ccwasccw = 0
    bias_othasoth = 0

    count_bias_cwascw = 0
    count_bias_ccwasccw = 0
    count_bias_othasoth = 0

    count_bias_cwasccw = 0
    count_bias_cwasoth = 0
    count_bias_ccwascw = 0
    count_bias_ccwasoth = 0
    count_bias_othascw = 0
    count_bias_othasccw = 0

    cw_number = 0
    ccw_number = 0
    oth_number = 0

    pred_cw_num = 0
    pred_ccw_num = 0
    pred_oth_num = 0

    error_index = []
    confusion_matrix_exp = []
    confusion_matrix_pred = []
    count_confusion_matrix_pred = []
    for i in range(expected.shape[0]):
        if expected[i] == 0:
            cw_number += 1
        elif expected[i] == 1:
            ccw_number += 1
        elif expected[i] == 2:
            oth_number += 1

        if result[i] == 0:
            pred_cw_num += 1
        elif result[i] == 1:
            pred_ccw_num += 1
        elif result[i] == 2:
            pred_oth_num += 1

        if result[i] != expected[i]:
            error_index.append([i, result[i], expected[i]])
            err += 1
            if expected[i] == 0:
                cw_err += 1
            elif expected[i] == 1:
                ccw_err += 1
            elif expected[i] == 2:
                oth_err += 1

            if result[i] == 0 and expected[i] == 1:
                bias_cwasccw += 1
            elif result[i] == 0 and expected[i] == 2:
                bias_cwasoth += 1
            elif result[i] == 1 and expected[i] == 0:
                bias_ccwascw += 1
            elif result[i] == 1 and expected[i] == 2:
                bias_ccwasoth += 1
            elif result[i] == 2 and expected[i] == 0:
                bias_othascw += 1
            elif result[i] == 2 and expected[i] == 1:
                bias_othasccw += 1
        elif result[i] == 0 and expected[i] == 0:
            bias_cwascw += 1
        elif result[i] == 1 and expected[i] == 1:
            bias_ccwasccw += 1
        elif result[i] == 2 and expected[i] == 2:
            bias_othasoth += 1

        if result_count[i] != expected[i]:
            err_count += 1
            if expected[i] == 0:
                cw_err_count += 1
            elif expected[i] == 1:
                ccw_err_count += 1
            elif expected[i] == 2:
                oth_err_count += 1

            if result_count[i] == 0 and expected[i] == 1:
                count_bias_cwasccw += 1
            elif result_count[i] == 0 and expected[i] == 2:
                count_bias_cwasoth += 1
            elif result_count[i] == 1 and expected[i] == 0:
                count_bias_ccwascw += 1
            elif result_count[i] == 1 and expected[i] == 2:
                count_bias_ccwasoth += 1
            elif result_count[i] == 2 and expected[i] == 0:
                count_bias_othascw += 1
            elif result_count[i] == 2 and expected[i] == 1:
                count_bias_othasccw += 1
        elif result_count[i] == 0 and expected[i] == 0:
            count_bias_cwascw += 1
        elif result_count[i] == 1 and expected[i] == 1:
            count_bias_ccwasccw += 1
        elif result_count[i] == 2 and expected[i] == 2:
            count_bias_othasoth += 1

        # confusion_matrix_pred.append([bias_cwascw, bias_cwasccw, bias_cwasoth])
        # confusion_matrix_pred.append([bias_ccwascw, bias_ccwasccw, bias_ccwasoth])
        # confusion_matrix_pred.append([bias_othascw, bias_othasccw, bias_othasoth])
        #
        # confusion_matrix_exp.append([bias_cwasoth+bias_cwascw+bias_cwasccw, 0, 0])
        # confusion_matrix_exp.append([0, bias_ccwasoth+bias_ccwascw+bias_ccwasccw, 0])
        # confusion_matrix_exp.append([0, 0, bias_othasccw+bias_othascw+bias_othasoth])
        #
        # count_confusion_matrix_pred.append([count_bias_cwascw, count_bias_cwasccw, count_bias_cwasoth])
        # count_confusion_matrix_pred.append([count_bias_ccwascw, count_bias_ccwasccw, count_bias_ccwasoth])
        # count_confusion_matrix_pred.append([count_bias_othascw, count_bias_othasccw, count_bias_othasoth])
        #
        # count_confusion_matrix_pred = np.asarray(count_confusion_matrix_pred)
        # confusion_matrix_exp = np.asarray(confusion_matrix_exp)
        # confusion_matrix_pred = np.asarray(confusion_matrix_pred)
    classes = [0, 1, 2]
    classes = np.asarray(classes, np.int8)
    # plot_confusion_matrix(result, expected, classes, normalize=True, title='{} bias confusion matrix'.format(idx))
    # plot_confusion_matrix(result_count, expected, classes, normalize=True, title='counted {} bias confusion matrix'.format(idx))

    pred_x = []
    exp_x = []

    pred_x.append(pred_cw_num)
    pred_x.append(pred_ccw_num)
    pred_x.append(pred_oth_num)

    exp_x.append(cw_number)
    exp_x.append(ccw_number)
    exp_x.append(oth_number)
    global total_cw
    global total_ccw
    global total_oth
    global pred_total_cw
    global pred_total_ccw
    global pred_total_oth
    total_cw += cw_number
    total_ccw += ccw_number
    total_oth += oth_number
    pred_total_cw += pred_cw_num
    pred_total_ccw += pred_ccw_num
    pred_total_oth += pred_oth_num

    for i in range(0, 3):
        exp_x.append(0)
        pred_x.insert(0, 0)

    x = np.arange(6)
    plt.bar(x, height=exp_x, color=(0.2, 0.4, 0.6, 0.6))
    plt.bar(x, height=pred_x)
    plt.xticks(x, ['CW', 'CCW', 'OTH', 'CW', 'CCW', 'OTH'])
    #
    # plt.bar(x, height=exp_x)
    # plt.xticks(x, ['CW', 'CCW', 'OTH'])
    plt.savefig('bias/bias_figs/{}_bias_hist.jpg'.format(idx))
    plt.clf()
    # plt.bar(x, height=pred_x)
    # plt.xticks(x, ['CW', 'CCW', 'OTH'])
    # plt.savefig('bias/{}_bias_hist_predicated.jpg'.format(idx))
    # plt.clf()



    error_index = np.asarray(error_index)
    np.savetxt("error_index/{}_error_index.txt".format(idx), error_index)

    print("VID {}: 24 - feature vector ACC: {}/{} = {} <-> CW_ERR {}, CCW_ERR {}, OTH_ERR {}".format(idx, err, expected.shape[0], 100 - err/expected.shape[0]*100, cw_err, ccw_err, oth_err))
    # print("BIAS TABLE:")
    # print("bias_cwasccw: {}/{} = {}".format(bias_cwasccw, cw_number, bias_cwasccw / cw_number * 100))
    # print("bias_cwasoth {}/{} = {}".format(bias_cwasoth, cw_number, bias_cwasoth / cw_number * 100))
    # print("bias_ccwascw: {}/{} = {}".format(bias_ccwascw, ccw_number, bias_ccwascw / ccw_number * 100))
    # print("bias_ccwasoth: {}/{} = {}".format(bias_ccwasoth, ccw_number, bias_ccwasoth / ccw_number * 100))
    # print("bias_othascw: {}/{} = {}".format(bias_othascw, oth_number, bias_othascw / oth_number * 100))
    # print("bias_othasccw: {}/{} = {}".format(bias_othasccw, oth_number, bias_othasccw / oth_number * 100))
    #
    print("VID {}: P/N/Z counts vector ACC: {}/{} = {} <-> CW_ERR {}, CCW_ERR {}, OTH_ERR {}".format(idx, err_count, expected.shape[0],100 - err_count/expected.shape[0]*100, cw_err_count, ccw_err_count, oth_err_count))
    # print("BIAS TABLE:")
    # print("bias_cwasccw: {}/{} = {}".format(count_bias_cwasccw, cw_number, count_bias_cwasccw / cw_number * 100))
    # print("bias_cwasoth {}/{} = {}".format(count_bias_cwasoth, cw_number, count_bias_cwasoth / cw_number * 100))
    # print("bias_ccwascw: {}/{} = {}".format(count_bias_ccwascw, ccw_number, count_bias_ccwascw / ccw_number * 100))
    # print("bias_ccwasoth: {}/{} = {}".format(count_bias_ccwasoth, ccw_number, count_bias_ccwasoth / ccw_number * 100))
    # print("bias_othascw: {}/{} = {}".format(count_bias_othascw, oth_number, count_bias_othascw / oth_number * 100))
    # print("bias_othasccw: {}/{} = {}".format(count_bias_othasccw, oth_number, count_bias_othasccw / oth_number * 100))


def trainer():
    train_vids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    labels = []
    trains = []
    idx = None
    for i in train_vids:
        if i in train_vids:
            if i < 10:
                idx = "0{}".format(i)
            else:
                idx = "{}".format(i)

        ls = np.loadtxt('loc_label/{}.txt'.format(idx))
        ys = ls[:, 2]
        xs = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt'.format(idx))
        print(ys.shape)
        print(xs.shape)
        if len(ys) != len(xs):
            print("{} Error count".format(idx))
            assert False
        for x in xs:
            trains.append(x)
        for y in ys:
            labels.append(y)

    labels = np.asarray(labels)
    trains = np.asarray(trains)

    np.savetxt('data05072019/trains.txt', trains)
    np.savetxt('data05072019/labels.txt', labels)

    trains = np.loadtxt('data05072019/trains.txt')
    labels = np.loadtxt('data05072019/labels.txt')
    # print(trains.shape)
    # print(labels.shape)
    simplified_x = []
    for x_row in trains:
        p_count = 0
        n_count = 0
        z_count = 0
        for x in x_row:
            if x > 0:
                p_count += 1
            elif x < 0:
                n_count += 1
            elif x == 0:
                z_count += 1
        simplified_x.append([p_count, n_count, z_count])
    simplified_x = np.asarray(simplified_x)
    np.savetxt('data05072019/simplified_x.txt', simplified_x)
    new_train = []  # cw and ccw train
    simplified_new_train = []  # simplified cw and ccw train
    new_label = []  # cw and ccw label
    rot_non_label = []  # rot_non label
    rot_non_train = []  # rot_non_train
    simplified_rot_non_train = []
    non_count = 0
    rot_count = 0
    for idx in range(0, trains.shape[0]):
        if labels[idx] != 2:
            rot_count += 1
            new_train.append(trains[idx])
            simplified_new_train.append(simplified_x[idx])
            new_label.append(labels[idx])
            rot_non_train.append(trains[idx])
            simplified_rot_non_train.append(simplified_x[idx])
            rot_non_label.append(0)  # CW & CW
        elif labels[idx] == 2:
            non_count += 1
            rot_non_train.append(trains[idx])
            simplified_rot_non_train.append(simplified_x[idx])
            rot_non_label.append(1)  # NON/ OTHER

    rot_non_label = np.asarray(rot_non_label)
    rot_non_train = np.asarray(rot_non_train)
    simplified_rot_non_train = np.asarray(simplified_rot_non_train)
    simplified_new_train = np.asarray(simplified_new_train)
    new_label = np.asarray(new_label)
    new_train = np.asarray(new_train)
    print()
    print(trains.shape)
    print()

    class_weight = {
        0: 1 - (rot_count / (rot_count + non_count)),
        1: 1 - (non_count / (rot_count + non_count))
    }
    # draw_histograms(rot_non_train, rot_non_label)
    svm_classifier(rot_non_train, rot_non_label, "g2", class_weight_=class_weight)
    svm_classifier(new_train, new_label, "g1")
    svm_classifier(simplified_x, rot_non_label, "h2", class_weight_=class_weight)
    svm_classifier(simplified_new_train, new_label, "h1")


def get_color(l):
    if l == 0:  # Blue CW
        color = (255, 0, 0)
    elif l == 1:  # Green CCW
        color = (0, 255, 0)
    else:  # Red Complex
        color = (0, 0, 255)
    return color


def mark_error(idx):
    error_index = np.loadtxt("ideal_train/error{}.txt".format(idx))
    locs = np.loadtxt("const_index/loc/{}.txt".format(idx))
    # error_index = error_index.tolist()
    imgs = []
    for j in range(25):
        img = cv2.imread('/Users/cheng_stark/tmp/rotation_results/XY{}_video/frame{}.jpg'.format(idx, j))
        for i in range(locs.shape[0]):
            for row in error_index:
                index, pred, exp = row
                if i == index:
                    x, y, w, h = locs[i]
                    print(x, y, w, h)
                    pred_color = get_color(pred)
                    exp_color = get_color(exp)
                    cv2.rectangle(img, (int(x - w / 2 - 8), int(y - h / 2 - 8)), (int(x + w / 2 + 8), int(y + h / 2 + 8)), exp_color, 2)
                    cv2.rectangle(img, (int(x - w / 2 - 0.5), int(y - h / 2 - 0.5)), (int(x + w / 2 + 0.5), int(y + h / 2 + 0.5)), pred_color, 2)
                    cv2.putText(img, 'Outer Box - Expected Label', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(img, 'Inner Box - Predicated Label', (380, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(img, 'CW', (800, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(img, 'CCW', (900, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, 'OTH', (1000, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        imgs.append(img)
    video_path = 'ideal_train/{}_error.avi'.format(idx)
    height, width = imgs[0].shape[0], imgs[0].shape[1]
    size = (width, height)
    fps = 6
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # video = cv2.VideoWriter(video_path, fourcc, fps, size)
    video = cv2.VideoWriter()
    video.open(video_path, fourcc, fps, size, True)
    id = 0
    for img in imgs:
        video.write(img)
        print('frame {} written!'.format(id))
        id += 1
    video.release()


def reverse_video(idx):
    print("Generating reverse for video {}".format(idx))
    imgs = []
    for j in range(25):
        img = cv2.imread('/Users/cheng_stark/tmp/rotation_results/XY{}_video/frame{}.jpg'.format(idx, j))
        imgs.insert(0, img)
    # imgs.reverse()
    video_path = 'reversed_videos/XY{}_reversed_video.avi'.format(idx)
    height, width = imgs[0].shape[0], imgs[0].shape[1]
    size = (width, height)
    fps = 6
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # video = cv2.VideoWriter(video_path, fourcc, fps, size)
    video = cv2.VideoWriter()
    video.open(video_path, fourcc, fps, size, True)
    id = 0
    for img in imgs:
        video.write(img)
        print('frame {} written!'.format(id))
        id += 1
    video.release()
    print("Generated reverse for video {}".format(idx))


def reverse_label(idx):
    label = np.loadtxt("const_index/lab/{}.txt".format(idx))
    rev_l = copy.deepcopy(label)
    rev_l = rev_l[::-1]
    np.savetxt("const_index/lab/{}_reversed.txt".format(idx), rev_l)


def draw_histograms(video_degrees, labels):
    for i in range(0, video_degrees.shape[0]):
        intensity = video_degrees[i, :]
        bins = np.arange(0, 24, 1)
        plt.figure()
        plt.plot(bins, intensity, marker='o', mec='r', mfc='w', label=u'y=x^2曲线图')
        plt.hlines(0, 24, 0.5, colors="c", linestyles="dashed")
        plt.xlim((0, 24))
        plt.ylim((-20, 20))
        plt.xlabel('Frames')
        plt.ylabel('Degree')
        plt.title('Cell {}: Rotation Degrees Label{}'.format(i, labels[i]))
        plt.savefig('bias/histograms/Cell{}.jpg'.format(i))
        print('{}/{}: histogram written'.format(i+1, video_degrees.shape[0]))


def cross_val():
    train_vids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    tmp = np.zeros((0, 25))
    for i in train_vids:
        if i < 10:
            video_id = "0{}".format(i)
        else:
            video_id = "{}".format(i)
        ls = np.loadtxt('loc_label/{}.txt'.format(video_id))
        ys = ls[:, 2]
        ys = np.asarray(ys)
        ys = np.reshape(ys, (-1, 1))
        xs = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt'.format(video_id))

        # creating reverse data
        xs_rev = xs[::-1]
        xs_rev = np.multiply(xs_rev, -1)
        ys_rev = []
        for l in ys:
            if l == 0:
                ys_rev.append(1)
            elif l == 1:
                ys_rev.append(0)
            elif l == 2:
                ys_rev.append(2)
        ys_rev = np.asarray(ys_rev)
        ys_rev = np.reshape(ys_rev, (-1, 1))
        print(xs.shape, ys.shape, xs_rev.shape, ys_rev.shape)
        x_y_rev = np.concatenate((xs_rev, ys_rev), axis=1)

        x_y = np.concatenate((xs, ys), axis=1)
        tmp = np.concatenate((tmp, x_y), axis=0)

        # insert rev
        # tmp = np.concatenate((tmp, x_y_rev), axis=0)

    np.savetxt("10_men/a.txt", tmp)
    for n in range(6):
        tmp = np.delete(tmp, 0, 0)
    print(tmp.shape)
    tmp = np.split(tmp, 10, axis=0)
    print(len(tmp))
    for validatio_id in range(10):
        current_validation = tmp[validatio_id]
        current_train = np.zeros((0, 25))
        for n in range(10):
            if n is not validatio_id:
                # print(current_train.shape, tmp[n].shape)
                current_train = np.concatenate((current_train, tmp[n]), axis=0)
        c_train = []
        c_valid = []

        for row in current_train:
            if row[24] == 0 or row[24] == 1:
                c_train.append(row)
        for row in current_validation:
            if row[24] == 0 or row[24] == 1:
                c_valid.append(row)
        current_validation = np.asarray(c_valid)
        current_train = np.asarray(c_train)
        current_validation_y = current_validation[:, 24]
        current_validation = np.delete(current_validation, 24, axis=1)
        current_validation_x = current_validation
        current_train_y = current_train[:, 24]
        current_train = np.delete(current_train, 24, axis=1)
        current_train_x = current_train
        print(current_train_x.shape, current_train_y.shape)
        # current_svm = sklearn.svm.LinearSVC(random_state=10, tol=1e-20, max_iter=500000, C=1)
        current_svm = sklearn.svm.SVC(kernel='linear', random_state=10, tol=1e-20, max_iter=500000, C=1, probability=True)
        # scalar = sklearn.preprocessing.StandardScaler()
        # current_train_x = scalar.fit_transform(current_train_x)
        # current_validation_x = scalar.fit_transform(current_validation_x)
        current_svm.fit(current_train_x, current_train_y)
        with open('10_men/mixed_models/{}.pickle'.format(validatio_id), 'wb') as f:
            pickle.dump(current_svm, f)
            print('{} SVM model saved!'.format(validatio_id))
        with open('10_men/mixed_models/{}.pickle'.format(validatio_id), 'rb') as f:
            current_svm = pickle.load(f)
        train_result = current_svm.predict(current_train_x)
        test_result = current_svm.predict(current_validation_x)
        train_acc = (current_train_y == train_result).sum() / float(current_train_y.size)
        test_acc = (current_validation_y == test_result).sum() / float(current_validation_y.size)
        print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))


def voting(feature_vec, threshold):
    # scalar = sklearn.preprocessing.StandardScaler()
    # feature_vec_ = scalar.fit_transform(feature_vec)
    cw_count = 0
    ccw_count = 0
    proba = [[0, 0]]
    proba = np.asarray(proba)
    for i in range(10):
        with open('10_men/single_model/{}.pickle'.format(i), 'rb') as f:
            current_svm = pickle.load(f)
        # feature_vec = np.reshape(feature_vec, (-1, 1))
        pred = current_svm.predict(feature_vec)
        proba = np.add(proba, current_svm.predict_proba(feature_vec))
        # print(proba)
        if pred == 0:
            cw_count += 1
        elif pred == 1:
            ccw_count += 1
    # print(proba)
    # print(cw_count, ccw_count)
    proba = np.divide(proba, 10)
    print(ccw_count, cw_count)
    if abs(ccw_count - cw_count) < threshold:
        # if threshold == 3:
            # print("GOT ONE")
        return 2, proba
    elif ccw_count > cw_count:
        return 1, proba
    elif cw_count > ccw_count:
        return 0, proba


def rev_voting(feature_vec, threshold):
    # scalar = sklearn.preprocessing.StandardScaler()
    # feature_vec_ = scalar.fit_transform(feature_vec)
    # scalar = sklearn.preprocessing.StandardScaler()
    # feature_vec_ = scalar.fit_transform(feature_vec)
    cw_count = 0
    ccw_count = 0
    proba = [[0, 0]]
    proba = np.asarray(proba)
    for i in range(10):
        with open('10_men/single_model/{}.pickle'.format(i), 'rb') as f:
            current_svm = pickle.load(f)
        # feature_vec = np.reshape(feature_vec, (-1, 1))
        pred = current_svm.predict(feature_vec)
        proba = np.add(proba, current_svm.predict_proba(feature_vec))
        # proba = [[0,0]]
        # print(proba)
        if pred == 0:
            cw_count += 1
        elif pred == 1:
            ccw_count += 1
    # print(proba)
    # print(cw_count, ccw_count)
    # proba = np.divide(proba, 10)
    if abs(ccw_count - cw_count) < threshold:
        # if threshold == 3:
        # print("GOT ONE")
        return 2, proba
    elif ccw_count > cw_count:
        return 1, proba
    elif cw_count > ccw_count:
        return 0, proba


def clean_data_testing(idx, thresh):
    vecs = np.loadtxt("/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt".format(idx))
    lab = np.loadtxt("const_index/lab/{}.txt".format(idx))
    exp_labs = []
    exp_cw = 0
    exp_ccw = 0
    exp_oth = 0
    for row in lab:
        l = row[2]
        if l == 0:
            exp_cw += 1
        elif l == 1:
            exp_ccw += 1
        elif l == 2:
            exp_oth += 1
        exp_labs.append(l)
    exp_labs = np.asarray(exp_labs)
    # exp_labs = np.reshape(exp_labs, (-1, 1))
    preds = []
    cw = 0
    ccw = 0
    oth = 0
    index = 0
    for row in vecs:
        row = np.reshape(row, (1, -1))
        pred, proba = voting(row, thresh)
        preds.append(pred)
        # if pred != exp_labs[index]:
        #     print("WRONG: {}<->{}".format(proba, np.std(proba)))
        # else:
        #     print("RIGHT: {}<->{}".format(proba, np.std(proba)))
        if pred == 0:
            cw += 1
        elif pred == 1:
            ccw += 1
        elif pred == 2:
            oth += 1
        index += 1
    pred_labs = np.asarray(preds)
    train_acc = (pred_labs == exp_labs).sum() / float(exp_labs.size)
    pred_x = []
    exp_x = []
    pred_x.append(cw)
    pred_x.append(ccw)
    pred_x.append(oth)
    exp_x.append(exp_cw)
    exp_x.append(exp_ccw)
    exp_x.append(exp_oth)
    for i in range(0, 3):
        exp_x.append(0)
        pred_x.insert(0, 0)

    x = np.arange(6)
    plt.bar(x, height=exp_x, color=(0.2, 0.4, 0.6, 0.6))
    plt.bar(x, height=pred_x)
    plt.xticks(x, ['CW', 'CCW', 'OTH', 'CW', 'CCW', 'OTH'])
    #
    # plt.bar(x, height=exp_x)
    # plt.xticks(x, ['CW', 'CCW', 'OTH'])
    plt.savefig('10_men/bias_figs/{}_bias_hist.jpg'.format(idx))
    plt.clf()
    print("Video {} acc: {}".format(idx, train_acc))

    err_cw = 0
    err_ccw = 0
    err_oth = 0
    cw_num = 0
    ccw_num = 0
    oth_num = 0
    for i in range(pred_labs.shape[0]):
        if exp_labs[i] == 0 and pred_labs[i] != 0:
            err_cw += 1
        elif exp_labs[i] == 1 and pred_labs[i] != 1:
            err_ccw += 1
        elif exp_labs[i] == 2 and pred_labs[i] != 2:
            err_oth += 1

        if exp_labs[i] == 0:
            cw_num += 1
        elif exp_labs[i] == 1:
            ccw_num += 1
        elif exp_labs[i] == 2:
            oth_num += 1

    calc = []
    calc.append(err_cw / cw_num)
    calc.append(err_ccw / ccw_num)
    calc.append(err_oth / oth_num)
    # print(calc)
    # print("Acc {} Stdev {}".format((pred_labs == exp_labs).sum() / float(exp_labs.size), statistics.stdev(calc)))
    return statistics.stdev(calc), 0
    # return (pred_labs == exp_labs).sum(), float(exp_labs.size)


def clean_data_testing_reverse(idx, thresh):
    vecs = np.loadtxt("/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt".format(idx))
    norm_vecs = np.loadtxt("/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt".format(idx))
    vecs = np.multiply(vecs, -1)
    lab = np.loadtxt("const_index/lab/{}.txt".format(idx))
    exp_labs = []
    exp_cw = 0
    exp_ccw = 0
    exp_oth = 0
    for row in lab:
        l = row[2]
        if l == 0:
            exp_cw += 1
        elif l == 1:
            exp_ccw += 1
        elif l == 2:
            exp_oth += 1
        exp_labs.append(l)
    exp_labs = np.asarray(exp_labs)
    # exp_labs = np.reshape(exp_labs, (-1, 1))
    preds = []
    cw = 0
    ccw = 0
    oth = 0

    index = 0
    for row in vecs:
        row = np.reshape(row, (1, -1))
        pred, proba = rev_voting(row, thresh)
        preds.append(pred)
        # if pred != exp_labs[index]:
        #     print("WRONG: {}<->{}".format(proba, np.std(proba)))
        # else:
        #     print("RIGHT: {}<->{}".format(proba, np.std(proba)))
        if pred == 0:
            cw += 1
        elif pred == 1:
            ccw += 1
        elif pred == 2:
            oth += 1
        index += 1
    pred_labs = np.asarray(preds)
    train_acc = (pred_labs == exp_labs).sum() / float(exp_labs.size)
    pred_x = []
    exp_x = []
    pred_x.append(cw)
    pred_x.append(ccw)
    pred_x.append(oth)
    # reversed for reverse videos
    exp_x.append(exp_ccw)
    exp_x.append(exp_cw)
    exp_x.append(exp_oth)
    for i in range(0, 3):
        exp_x.append(0)
        pred_x.insert(0, 0)

    x = np.arange(6)
    plt.bar(x, height=exp_x, color=(0.2, 0.4, 0.6, 0.6))
    plt.bar(x, height=pred_x)
    plt.xticks(x, ['CW', 'CCW', 'OTH', 'CW', 'CCW', 'OTH'])
    #
    # plt.bar(x, height=exp_x)
    # plt.xticks(x, ['CW', 'CCW', 'OTH'])
    plt.savefig('10_men/bias_figs/{}_bias_hist.jpg'.format(idx))
    plt.clf()
    print("Video {} acc: {}".format(idx, train_acc))

    err_cw = 0
    err_ccw = 0
    err_oth = 0
    cw_num = 0
    ccw_num = 0
    oth_num = 0
    for i in range(pred_labs.shape[0]):
        if exp_labs[i] == 0 and pred_labs[i] != 0:
            err_cw += 1
        elif exp_labs[i] == 1 and pred_labs[i] != 1:
            err_ccw +=1
        elif exp_labs[i] == 2 and pred_labs[i] != 2:
            err_oth += 1

        if exp_labs[i] == 0:
            cw_num += 1
        elif exp_labs[i] == 1:
            ccw_num += 1
        elif exp_labs[i] == 2:
            oth_num += 1

    calc = []
    calc.append(err_cw/cw_num)
    calc.append(err_ccw/ccw_num)
    calc.append(err_oth/oth_num)
    print(calc)
    print("Acc {} Stdev {}".format((pred_labs == exp_labs).sum()/float(exp_labs.size), statistics.stdev(calc)))
    return statistics.stdev(calc), 0

    # return (pred_labs == exp_labs).sum(), float(exp_labs.size)


def rev_ver(idx):
    norm = np.loadtxt("/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt".format(idx))
    # norm = np.loadtxt("10_men/ori.txt")
    rev = np.loadtxt("/Users/cheng_stark/tmp/rotation_results/XY{}_reversed_video/vectors.txt".format(idx))
    # rev = np.loadtxt("10_men/rev.txt")
    rev = np.fliplr(rev)
    for i in range(norm.shape[0]):
        print(norm[i])
        print(rev[i])
        print("-------------------------------------------------------------------------------------------------------")


def compare_vec(idx):
    norm1 = np.loadtxt("/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt".format(idx))
    norm2 = np.loadtxt("10_men/ori.txt")
    # rev = np.loadtxt("/Users/cheng_stark/tmp/rotation_results/XY{}_reversed_video/vectors.txt".format(idx))
    # rev = np.loadtxt("10_men/rev.txt")
    # rev = np.fliplr(rev)
    for i in range(norm1.shape[0]):
        print(norm1[i])
        print(norm2[i])
        print("-------------------------------------------------------------------------------------------------------")


def counting_compare(idx):
    vec = np.loadtxt("/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt".format(idx))
    counts = []
    label = np.loadtxt("const_index/lab/{}.txt".format(idx))
    for i in range(label.shape[0]):
        p = 0
        n = 0
        z = 0
        l = None
        if label[i][2] == 0:
            l = 'CW'
        elif label[i][2] == 1:
            l = 'CCW'
        elif label[i][2] == 2:
            l = 'OTH'
        for x in vec[i]:
            if x > 0:
                p += 1
            elif x < 0:
                n += 1
            elif x == 0:
                z += 1
        print('P: {} N: {} Z:{} | LABEL: {}'.format(p, n, z, l))
        counts.append([p, n, z])


def last_frame_patch(idx):
    locs = np.loadtxt("const_index/loc/{}.txt".format(idx))
    frame = cv2.imread("/Users/cheng_stark/tmp/rotation_results/XY{}_video/frame24.jpg".format(idx))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(frame.shape)
    cell_id = 0
    for row in locs:
        x, y, w, h = row
        print(x, y)
        patch = frame[int(y - 52/2):int(y + 52/2), int(x - 52/2):int(x + 52/2)]
        cv2.imwrite("/Users/cheng_stark/tmp/rotation_results/XY{}_video/patch/24_{}.jpg".format(idx, cell_id), patch)
        print('Saved {}'.format(cell_id))
        cell_id += 1


def translate_label(x):
    if x == 0:
        return 'CW'
    elif x == 1:
        return 'CCW'
    elif x == 2:
        return 'OTH/CPLX'


def sample100(idx):
    global sample100_id
    correct_label = np.loadtxt("loc_label/{}.txt".format(idx))
    error_record = np.loadtxt("error_index/{}_error_index.txt".format(idx))
    locs = np.loadtxt("const_index/loc/{}.txt".format(idx))
    vectors = np.loadtxt("/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt".format(idx))
    error_i = []
    for row in error_record:
        index, pred, exp = row
        error_i.append(index)
    for i in range(locs.shape[0]):
        if sample100_id <= 99:
            if i in error_i:
                vec = vectors[i]
                cell_patches = []
                blank = np.zeros((480, 1040))
                blank.fill(255)
                blank[0, 0] = 0
                cell_patches.append(blank)
                for frame_id in range(25):
                    patch = cv2.imread("/Users/cheng_stark/tmp/rotation_results/XY{}_video/patch/{}_{}.jpg".format(idx, frame_id, i))
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    patch = cv2.resize(patch, (104, 104))
                    back = np.zeros((480, 1040))
                    y_offset = 50
                    x_offset = 30
                    back[y_offset:y_offset + patch.shape[0], x_offset:x_offset + patch.shape[1]] = patch
                    vec_list = vec.tolist()
                    for x in range(len(vec_list)):
                        vec_list[x] = int(vec_list[x])
                    cv2.putText(back, '{}'.format(vec_list), (30, 300), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(back, 'Current frame index: {}'.format(frame_id), (300, 80),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                    for row in error_record:
                        error_idx, pred, exp = row
                        if error_idx == i:
                            cv2.putText(back, 'Human label:     {}'.format(translate_label(exp)), (300, 120),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                            cv2.putText(back, 'Predicted label:   {}'.format(translate_label(pred)), (300, 140),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                    if frame_id < 24:
                        cv2.putText(back, 'Rotation angle to the next frame: {}'.format(int(vec[frame_id])), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                    else:
                        cv2.putText(back, 'N/A', (300, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                    (255, 0, 0), 2)
                    cell_patches.append(back)
                imageio.mimsave('sample100/marked/error/{}.gif'.format(sample100_id), cell_patches, duration=1/3)
                print('Saved gif {}'.format(sample100_id))
                # vec = vec.tolist()
                # vec = np.asarray(vec, dtype='int64')
                # np.savetxt('sample100/marked/error/{}.txt'.format(sample100_id), vec, fmt="%d")
                sample100_id += 1
            else:
                vec = vectors[i]
                cell_patches = []
                blank = np.zeros((480, 1040))
                blank.fill(255)
                blank[0, 0] = 0
                cell_patches.append(blank)
                for frame_id in range(25):
                    patch = cv2.imread("/Users/cheng_stark/tmp/rotation_results/XY{}_video/patch/{}_{}.jpg".format(idx, frame_id, i))
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    patch = cv2.resize(patch, (104, 104))
                    back = np.zeros((480, 1040))
                    y_offset = 50
                    x_offset = 30
                    back[y_offset:y_offset + patch.shape[0], x_offset:x_offset + patch.shape[1]] = patch
                    vec_list = vec.tolist()
                    for x in range(len(vec_list)):
                        vec_list[x] = int(vec_list[x])
                    cv2.putText(back, '{}'.format(vec_list), (30, 300), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(back, 'Current frame index: {}'.format(frame_id), (300, 80),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                    cv2.putText(back, 'Human label:     {}'.format(translate_label(correct_label[i][2])), (300, 120),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                    cv2.putText(back, 'Predicted label:   {}'.format(translate_label(correct_label[i][2])), (300, 140),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                    if frame_id < 24:
                        cv2.putText(back, 'Rotation angle to the next frame: {}'.format(int(vec[frame_id])), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                    else:
                        cv2.putText(back, 'N/A', (300, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                    (255, 0, 0), 2)
                    cell_patches.append(back)
                imageio.mimsave('sample100/marked/correct/{}.gif'.format(sample100_id), cell_patches, duration=1/3)
                print('Saved gif {}'.format(sample100_id))
                # vec = vec.tolist()
                # vec = np.asarray(vec, dtype='int64')
                # np.savetxt('sample100/marked/correct/{}.txt'.format(sample100_id), vec, fmt="%d")
                sample100_id += 1


def feature_vector_filtering():
    all_dict = dict()
    for i in range(0, 11):
        # thresh = int(24/10*i)
        all_dict[i/10] = 0

    print(all_dict.keys())
    for index in train_set_index:
        if index < 10:
            index = '0{}'.format(index)
        vecs = np.loadtxt('/Users/cheng_stark/tmp/rotation_results_stable/XY{}_video/vectors.txt'.format(index))
        label = np.loadtxt("const_index/lab/{}.txt".format(index))
        cell_id = 0
        for vec in vecs:
            # CW 0 <0
            # CCW 1 >0
            # OTH bla
            cell_label = label[cell_id][2]
            pos = 0
            neg = 0
            z = 0
            sum = 0
            for x in vec:
                if x > 0:
                    pos += 1
                elif x < 0:
                    neg += 1
                elif x == 0:
                    z += 1
                sum += 1

            pos_perc = int(pos/sum*10) / 10
            neg_perc = int(neg/sum*10) / 10
            print(pos_perc, neg_perc, cell_label)
            if cell_label == 0:
                all_dict[neg_perc] += 1
            elif cell_label == 1:
                all_dict[pos_perc] += 1
            cell_id += 1

    y_pos = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    s = 0
    for x in all_dict.keys():
        if x >= 0.8:
            s += all_dict[x]
    print(s)
    # Create bars
    plt.bar(y_pos, all_dict.values())

    # Create names on the x-axis
    plt.xticks(y_pos, all_dict.keys())

    plt.savefig('filtering/fig.jpg')

    # Show graphic
    plt.show()


def feature_vector_filtering_per_vid(idx):
    all_dict = dict()
    for i in range(0, 11):
        # thresh = int(24/10*i)
        all_dict[i/10] = 0

    print(all_dict.keys())
    vecs = np.loadtxt('/Users/cheng_stark/tmp/rotation_results_stable/XY{}_video/vectors.txt'.format(idx))
    # vec2 = []
    # for i in range(vecs.shape[0]):
    #     arow = []
    #     for j in range(vecs[i].shape[0]):
    #         if j % 2 == 0:
    #             arow.append(vecs[i][j])
    #     arow = np.asarray(arow)
    #     vec2.append(arow)
    # vecs = np.asarray(vec2)
    label = np.loadtxt("const_index/lab/{}.txt".format(idx))
    cell_id = 0
    for vec in vecs:
        # CW 0 <0
        # CCW 1 >0
        # OTH bla
        cell_label = label[cell_id][2]
        pos = 0
        neg = 0
        z = 0
        sum = 0
        for x in vec:
            if x > 0:
                pos += 1
            elif x < 0:
                neg += 1
            elif x == 0:
                z += 1
            sum += 1

        pos_perc = int(pos/sum*10) / 10
        neg_perc = int(neg/sum*10) / 10
        if cell_label != 2:
            print(pos_perc, neg_perc, cell_label)
        if cell_label == 0:
            all_dict[neg_perc] += 1
        elif cell_label == 1:
            all_dict[pos_perc] += 1
        cell_id += 1

    y_pos = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    s = 0
    for x in all_dict.keys():
        if x >= 0.8:
            s += all_dict[x]
    print(s)
    # Create bars
    plt.bar(y_pos, all_dict.values())

    # Create names on the x-axis
    plt.xticks(y_pos, all_dict.keys())

    plt.savefig('filtering/fig_np_process{}.jpg'.format(idx))

    # Show graphic
    plt.show()


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        if y_value > 0:
            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.1f}".format(y_value)

            # Create annotation
            rect.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.


def plot_compare_filetrs(idx, thresh, thresh2):
    YES1 = 0
    YES2 = 0
    NO1 = 0
    NO2 = 0
    all_dict = dict()
    for i in range(0, 11):
        # thresh = int(24/10*i)
        all_dict[i / 10] = 0

    vecs = np.loadtxt('/Users/cheng_stark/tmp/rotation_results_stable/XY{}_video/vectors.txt'.format(idx))
    label = np.loadtxt("const_index/lab/{}.txt".format(idx))
    cell_id = 0
    for vec in vecs:
        # CW 0 <0
        # CCW 1 >0
        # OTH bla
        cell_label = label[cell_id][2]
        pos = 0
        neg = 0
        z = 0
        sum_ = 0
        for x in vec:
            if x > 0:
                pos += 1
            elif x < 0:
                neg += 1
            elif x == 0:
                z += 1
            sum_ += 1

        ratio = None
        if cell_label == 0:
            if pos == 0:
                ratio = 999
            else:
                ratio = neg / pos
            if ratio >= thresh2:
                YES1 += 1
                print(pos, neg)

            elif ratio <= 1:
                NO1 += 1
        elif cell_label == 1:
            if neg == 0:
                ratio = 999

            else:
                ratio = pos / neg
            if ratio >= thresh2:
                print(pos, neg)

                YES1 += 1
            elif ratio <= 1:
                NO1 += 1

        pos_perc = int(pos / sum_ * 10) / 10
        neg_perc = int(neg / sum_ * 10) / 10
        if cell_label != 2:
            # print(pos_perc, neg_perc, cell_label, ratio)
            pass
        if cell_label == 0:
            all_dict[neg_perc] += 1
        elif cell_label == 1:
            all_dict[pos_perc] += 1
        cell_id += 1

    x_pos_ = np.arange(0, 44, step=2)
    x_pos = []
    for x in x_pos_:
        x_pos.append(x)
    print(x_pos)
    s = 0
    for x in all_dict.keys():
        if x >= thresh:
            s += all_dict[x]
# ----------------------------------------------------------------------------------------------------------------- #
    bad = []
    all_dict2 = dict()
    for i in range(0, 11):
        # thresh = int(24/10*i)
        all_dict2[i / 10] = 0

    vecs2 = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt'.format(idx))

    label2 = np.loadtxt("const_index/lab/{}.txt".format(idx))
    cell_id = 0
    for vec in vecs2:
        line = []
        for i in range(vec.shape[0]):
            if i % 2 == 0:
                line.append(vec[i])
        # CW 0 <0
        # CCW 1 >0
        # OTH bla
        cell_label = label2[cell_id][2]
        pos = 0
        neg = 0
        z = 0
        sum2_ = 0
        for x in line:
            if x > 0:
                pos += 1
            elif x < 0:
                neg += 1
            elif x == 0:
                z += 1
            sum2_ += 1

        pos_perc = int(pos / sum2_ * 10) / 10
        neg_perc = int(neg / sum2_ * 10) / 10

        ratio = None
        if cell_label == 0:
            if pos == 0:
                ratio = 999
            else:
                ratio = neg / pos
            if ratio >= thresh2:
                print(pos, neg)
                YES2 += 1
            elif ratio <= 1:
                NO2 += 1
        elif cell_label == 1:
            if neg == 0:
                ratio = 999
            else:
                ratio = pos / neg
            if ratio >= thresh2:
                print(pos, neg)

                YES2 += 1
            elif ratio <= 1:
                NO2 += 1

        if pos_perc == 0.5 or neg_perc == 0.5:
            bad.append(cell_id)
        if cell_label != 2:
            # print('-------------')
            # print(pos, neg, cell_label, ratio)
            pass
        if cell_label == 0:
            all_dict2[neg_perc] += 1
        elif cell_label == 1:
            all_dict2[pos_perc] += 1
        cell_id += 1

    # s2 = 0
    # for x in all_dict2.keys():
    #     if x >= thresh:
    #         s2 += all_dict2[x]
    # print('{} - {}'.format(s, s2))

    print('ORI: {} - {}, AFT: {} - {}'.format(YES1, NO1, YES2, NO2))

    x2 = []
    for x in all_dict2.values():
        x2.append(x)
    x1 = []
    for x in all_dict.values():
        x1.append(x)

    xx = x1 + x2
    for i, v in enumerate(xx):
        plt.text(x_pos[i] - 0.25, v + 0.01, str(v))

    for i in range(11):
        x2.insert(0, 0)

    for i in range(11):
        x1.append(0)

    print(len(x1), len(x_pos))
    bar1 = plt.bar(x_pos, x1)
    bar2 = plt.bar(x_pos, x2)
    for x in range(len(bar2)):
        bar2[x].set_color('r')

    keys = []
    for x in all_dict.keys():
        keys.append(x)

    for x in all_dict.keys():
        keys.append(x)

    assert sum(all_dict.values()) == sum(all_dict2.values())
    plt.xticks(x_pos, keys)

    red_patch = mpatches.Patch(color='red', label='VEC with added process')
    blue_patch = mpatches.Patch(color='#1f77b4', label='VEC without added process')

    plt.legend(handles=[red_patch, blue_patch])

    plt.savefig('filtering/fig_both{}.jpg'.format(idx))

    # Show graphic
    plt.show()

    return idx, bad


def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode


def feature_vector_filtering_per_vid_by_ssim(idx):
    all_dict = dict()
    for i in range(0, 11):
        # thresh = int(24/10*i)
        all_dict[i/10] = 0

    # CW 0 <0
    # CCW 1 >0
    # OTH bla
    vecs = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt'.format(idx))
    label = np.loadtxt("const_index/lab/{}.txt".format(idx))
    cell_id = 0
    yes_count = 0
    oth_offset = 0
    for row in vecs:
        line = ''
        p = 0
        n = 0
        z = 0
        ssims = []
        for x in range(len(row)):
            if x % 2 == 1:
                ssims.append(round(row[x], 2))

        print('Mode: {}'.format(find_max_mode(ssims)))
        print('Meann: {}'.format(statistics.mean(ssims)))

        # print(ssims)
        for x in range(len(row)):
            if x % 2 == 1:
                if row[x] >= find_max_mode(ssims):
                # if row[x] >= statistics.mean(ssims):
                # if row[x] >= 0:
                #     line += '{} '.format(row[x-1])
                    if row[x-1] > 0: p+=1
                    elif row[x-1] <0: n+=1
                    elif row[x-1] == 0: z+=1
                # else:
                #     line += 'X '
        line += ' -> '
        line += str(label[cell_id][2])
        line += ' |+{} -{} {} '.format(p, n, z)

        if label[cell_id][2] == 0 and p < n:
            line += ' YES '
            yes_count += 1
        elif label[cell_id][2] == 1 and p > n:
            line += ' YES '
            yes_count += 1
        elif label[cell_id][2] == 2 and p == n:
            line += ' YES '
            oth_offset += 1
        print(line)
        cell_id += 1
    print(' {}/ {} = {}'.format(yes_count, label.shape[0] - oth_offset, yes_count/(label.shape[0] - oth_offset)))


def mark_non_ideal(idx, non_ideas):
    os.makedirs('filtering/patch/{}'.format(idx))
    locs = np.loadtxt("const_index/loc/{}.txt".format(idx))
    label = np.loadtxt("const_index/lab/{}.txt".format(idx))
    # error_index = error_index.tolist()
    imgs = []
    for j in range(25):
        img = cv2.imread('/Users/cheng_stark/tmp/rotation_results_stable/XY{}_video/frame{}.jpg'.format(idx, j))
        for i in non_ideas:
            x, y, w, h = locs[i]
            color = get_color(int(label[i][2]))
            im = img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
            cv2.imwrite('filtering/patch/{}/{}_{}.jpg'.format(idx, j, i), im)
            cv2.rectangle(img, (int(x - w / 2 - 0.5), int(y - h / 2 - 0.5)),
                          (int(x + w / 2 + 0.5), int(y + h / 2 + 0.5)), color, 2)
            cv2.putText(img, '{}'.format(i), (int(x - w/2-2), int(y-h/2-2)), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
            cv2.putText(img, 'CW', (800, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img, 'CCW', (900, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, 'OTH', (1000, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        imgs.append(img)
    video_path = 'filtering/marked_video/{}_non_ideal.avi'.format(idx)
    height, width = imgs[0].shape[0], imgs[0].shape[1]
    size = (width, height)
    fps = 6
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # video = cv2.VideoWriter(video_path, fourcc, fps, size)
    video = cv2.VideoWriter()
    video.open(video_path, fourcc, fps, size, True)
    id = 0
    for img in imgs:
        video.write(img)
        print('frame {} written!'.format(id))
        id += 1
    video.release()


def video_vertical_flip(idx):
    imgs = []
    norm = []
    for j in range(25):
        img = cv2.imread('/Users/cheng_stark/tmp/rotation_results_stable/XY{}_video/frame{}.jpg'.format(idx, j))
        norm.append(img)
        img = cv2.flip(img, 1)
        imgs.append(img)
    video_path = 'filtering/flipped_vids/XY{}_flip.avi'.format(idx)
    video_path_norm = 'filtering/flipped_vids/XY{}_norm.avi'.format(idx)
    height, width = imgs[0].shape[0], imgs[0].shape[1]
    size = (width, height)
    fps = 6
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc1 = cv2.VideoWriter_fourcc(*'MJPG')
    # video = cv2.VideoWriter(video_path, fourcc, fps, size)
    video = cv2.VideoWriter()
    video.open(video_path, fourcc, fps, size, True)
    id = 0
    for img in imgs:
        video.write(img)
        print('frame {} written!'.format(id))
        id += 1
    video.release()

    video1 = cv2.VideoWriter()
    video1.open(video_path_norm, fourcc1, fps, size, True)
    id = 0
    for img in norm:
        video1.write(img)
        print('frame {} written!'.format(id))
        id += 1
    video1.release()


def flip_compare(idx):
    norm = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_norm/vectors.txt'.format(idx))
    flipped = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_flip/vectors.txt'.format(idx))
    for i in range(norm.shape[0]):
        norm_row = []
        flip_row = []
        for j in range(norm[i].shape[0]):
            if j % 2 == 0:
                norm_row.append(norm[i][j])
                flip_row.append(flipped[i][j]*(-1))
        print(norm_row)
        print(flip_row)
        print('-------------------------------------------------------')


def generate_flipped_loc(idx):
    img = cv2.imread('/Users/cheng_stark/tmp/rotation_results_stable/XY{}_video/frame{}.jpg'.format(idx, 0))
    length = img.shape[1] / 2
    print(length)
    loc = np.loadtxt("const_index/loc/{}.txt".format(idx))
    for i in range(loc.shape[0]):
        diff = length - loc[i][0]
        newloc = length + diff
        loc[i][0] = newloc
    np.savetxt('filtering/flipped_vids/{}.txt'.format(idx), loc)


def test():
    sample1 = cv2.imread('/Users/cheng_stark/tmp/rotation_results_stable/XY15_video/patch/0_0.jpg')
    sample2 = cv2.imread('/Users/cheng_stark/tmp/rotation_results_stable/XY15_video/patch/0_0.jpg')
    row, col = 52, 52
    rot_mat1 = cv2.getRotationMatrix2D((row / 2, col / 2), -180, 1.0)
    result1 = cv2.warpAffine(sample1, rot_mat1, (col, row), cv2.INTER_NEAREST)
    rot_mat2 = cv2.getRotationMatrix2D((row / 2, col / 2), 180, 1.0)
    result2 = cv2.warpAffine(sample2, rot_mat2, (col, row), cv2.INTER_NEAREST)
    print(np.array_equal(result1, result2))


def loc_check(idx):
    img = cv2.imread('/Users/cheng_stark/tmp/rotation_results/XY{}_flipped/frame0.jpg'.format(idx))
    loc = np.loadtxt('filtering/flipped_vids/{}.txt'.format(idx))
    print(img.shape)
    for row in loc:
        x, y, w, h = row
        cv2.rectangle(img, (int(x - w / 2 - 8), int(y - h / 2 - 8)), (int(x + w / 2 + 8), int(y + h / 2 + 8)), (255, 0, 0), 2)
        # cv2.rectangle(img, (1430, 710), (1440, 720), (255, 0, 0), 2)
    cv2.imwrite('filtering/flip_loc_check{}.jpg'.format(idx), img)


def frames_check():
    for i in range(25):
        norm_f = np.loadtxt('god/norm_frames/frame{}.txt'.format(i))
        flip_f = np.loadtxt('god/flip_frames/frame{}.txt'.format(i))
        flip_f = np.flip(flip_f, 1)
        print(np.array_equal(norm_f, flip_f))


def patches_check():
    degree_range = 5
    degrees = np.arange(-degree_range, degree_range + 1, 1)
    for i in range(67):
        for j in range(24):
            for degree in degrees:
                if degree == 0:
                    norm_p = np.loadtxt('god/15_patch_norm/{}_{}_{}.txt'.format(degree, j, i))
                    # cv2.imwrite('god/norm_patch/template_{}_{}.jpg'.format(j, i), norm_p)
                    flip_p = np.loadtxt('god/15_patch_flip/{}_{}_{}.txt'.format(degree, j, i))
                    flip_p = np.flip(flip_p, 1)
                    # cv2.imwrite('god/flip_patch/template_{}_{}.jpg'.format(j, i), flip_p)

                    ssim_index = ssim(norm_p, flip_p, data_range=flip_p.max() - flip_p.min())
                    print(np.array_equal(norm_p, flip_p), ssim_index, degree)


def data_loss_check(idx):
    loc = np.loadtxt('const_index/loc/{}.txt'.format(idx))
    loss = []
    loss_dict = dict()
    degree_range = 5
    degrees = np.arange(-degree_range, degree_range + 1, 1)
    for degree in degrees:
        loss_dict[degree] = []
    for i in range(loc.shape[0]):
        for j in range(24):
            zero_patch = np.loadtxt('data_loss/{}_patch/{}_{}_{}.txt'.format(idx, 0, j, i))
            blank_zero = np.zeros_like(zero_patch)
            blank_zero = cv2.circle(blank_zero, (26, 26), 25, 255, cv2.FILLED)
            blank_zero[blank_zero == 255] = zero_patch[blank_zero == 255]
            for degree in degrees:
                if degree != 0:
                    norm_p = np.loadtxt('data_loss/{}_patch/{}_{}_{}.txt'.format(idx, degree, j, i))
                    blank_p = np.zeros_like(norm_p)
                    blank_p = cv2.circle(blank_p, (26, 26), 25, 255, cv2.FILLED)
                    blank_p[blank_p == 255] = norm_p[blank_p == 255]
                    diff = abs(np.sum(blank_zero) - np.sum(blank_p))
                    loss.append(diff)
                    print(diff, degree)
                    loss_dict[degree].append(diff)
                else:
                    loss_dict[degree].append(0)
    print(statistics.mean(loss))
    loss_vs_degree = []
    for value in loss_dict.values():
        print(value)
        mean = statistics.mean(value)
        perc = mean / (52*52)*100
        perc = round(perc, 2)
        loss_vs_degree.append(perc)

    x_pos = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    for i, v in enumerate(loss_vs_degree):
        plt.text(x_pos[i] - 0.25, v + 0.01, str(v))

    bar1 = plt.bar(x_pos, loss_vs_degree)
    plt.xticks(x_pos,loss_dict.keys())
    plt.savefig('data_loss/loss_vs_degree{}.jpg'.format(idx))

    # Show graphic
    plt.show()


def label1645():
    alll = []
    for i in range(1, 33):
        if i < 10:
            i = '0{}'.format(i)
        ls = np.loadtxt('const_index/lab/{}.txt'.format(i))
        for row in ls:
            l = row[2]
            alll.append(l)
        print(i)
    alll = np.asarray(alll)
    print(alll.shape)
    np.savetxt('const_index/lab/all1645.txt', alll)


def handle_row(row, old=False):
    if not old:
        new_row = []
        p = 0
        n = 0
        z = 0

        for j in range(row.shape[0]):
            x = row[j]
            if j % 2 == 0:
                if x > 0:
                    p += 1
                elif x < 0:
                    n += 1
                elif x == 0:
                    z += 1
                new_row.append(x)
        new_row = np.asarray([new_row])
        return p, n, z, new_row

    elif old:
        p = 0
        n = 0
        z = 0

        for j in range(row.shape[0]):
            x = row[j]
            if x > 0:
                p += 1
            elif x < 0:
                n += 1
            elif x == 0:
                z += 1
        return p, n, z, row


def handle_row_rev(row, old=False):
    if not old:
        new_row = []
        p = 0
        n = 0
        z = 0

        for j in range(row.shape[0]):
            x = row[j]
            if j % 2 == 0:
                if x > 0:
                    p += 1
                elif x < 0:
                    n += 1
                elif x == 0:
                    z += 1
                new_row.append(x)
        new_row.reverse()
        new_row = np.asarray([new_row])
        new_row = np.multiply(new_row, -1)
        return n, p, z, new_row

    elif old:
        p = 0
        n = 0
        z = 0

        for j in range(row.shape[0]):
            x = row[j]
            if x > 0:
                p += 1
            elif x < 0:
                n += 1
            elif x == 0:
                z += 1
        row = row[::-1]
        return n, p, z, row


def filter_ideal_generate(thresh):
    cw_dict = dict()
    ccw_dict = dict()
    cw_count = 0
    ccw_count = 0
    ideal_vectors = []
    ideal_labels = []
    for idx in train_set_index:
        if idx < 10:
            idx = '0{}'.format(idx)
        vecs = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt'.format(idx))
        label = np.loadtxt('const_index/lab/{}.txt'.format(idx))
        for i in range(vecs.shape[0]):
            row = vecs[i]
            cell_label = label[i][2]
            p = 0
            n = 0
            z = 0

            for j in range(row.shape[0]):
                x = row[j]
                if j % 2 == 0:
                    if x > 0:
                        p += 1
                    elif x < 0:
                        n += 1
                    elif x == 0:
                        z += 1
            IDEAL = False
            if cell_label == 0:
                if p == 0 and n != 0:
                    ratio = 999
                elif n == 0 and p == 0:
                    ratio = 0
                else:
                    ratio = n / p
                if ratio >= thresh:
                    cw_count += 1
                    IDEAL = True
            elif cell_label == 1:
                if n == 0 and p != 0:
                    ratio = 999
                elif n == 0  and p == 0:
                    ratio = 0
                else:
                    ratio = p / n
                if ratio >= thresh:
                    ccw_count += 1
                    IDEAL = True

            if IDEAL:
                print('{} {} {} -> {}'.format(p, n, z, cell_label))
                row = np.asarray([p, n, z])
                ideal_vectors.append(row)
                ideal_labels.append(cell_label)

        print('Saved {}'.format(idx))

    ideal_vectors = np.asarray(ideal_vectors)
    ideal_labels = np.asarray(ideal_labels)

    print(ideal_labels.shape, ideal_vectors.shape)

    np.savetxt('ideal_train/new_train.txt', ideal_vectors)
    np.savetxt('ideal_train/new_label.txt', ideal_labels)

    print(cw_count, ccw_count)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def ideal_test(idx, thresh=None, vote=False):
    wrong = 0
    wrongidx = []
    with open('ideal_train/{}.pickle'.format('gen_new_CW_vs_CCW'), 'rb') as f:
        direction_svm = pickle.load(f)
    with open('ideal_train/{}.pickle'.format('f2'), 'rb') as f:
        rot_non_svm = pickle.load(f)
    video_ves = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt'.format(idx))
    exp_label = np.loadtxt('const_index/lab/{}.txt'.format(idx))
    y_true = []
    y_pred = []
    # 0 ROT || 1 NON
    for i in range(video_ves.shape[0]):
        exp = exp_label[i][2]
        p, n, z, row = handle_row(video_ves[i], old=False)
        ori_row = row.reshape((1, -1))
        row_for_pred = np.asarray([p, n, z]).reshape((1, -1))
        if vote:
            pred, proba = voting(ori_row, threshold=3)
        elif thresh is None and not vote:
            # pred = direction_svm.predict(row_for_pred)
            # proba = direction_svm.predict_proba(row_for_pred)
            # if statistics.stdev(proba[0]) <= 0.15:
            #     pred = 2
            rot_non_pred = rot_non_svm.predict(row_for_pred)
            print(rot_non_pred)

            if rot_non_pred == 1:
                pred = 2
            else:
                pred = direction_svm.predict(row_for_pred)
                proba = direction_svm.predict_proba(row_for_pred)
                if statistics.stdev(proba[0]) <= 0.5:
                    pred = 2
        else:
            if p > n:
                if n == 0:
                    pred = 1
                else:
                    if p / n >= thresh:
                        pred = 1
                    else:
                        pred = 2
            elif n > p:
                if p == 0:
                    pred = 0
                else:
                    if n / p >= thresh:
                        pred = 0
                    else:
                        pred = 2
            else:
                pred = 2

        if exp != pred:
            wrongidx.append([i, int(pred), int(exp)])
            wrong += 1
            if thresh is None:
                # print('Cell_index {} EXP: {}, PRED: {}, PROBA: {}-{:.2f}, Counts: {}'.format(i, exp, pred, proba, statistics.stdev(proba[0]), (p, n, z)))
                pass
        y_pred.append(pred)
        y_true.append(exp)
    # print('ACC: {}, xOTHx {}'.format((video_ves.shape[0] - wrong) / video_ves.shape[0], wrong_oth))
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    # print(y_true)
    # print(y_pred)
    plot_confusion_matrix(y_true, y_pred, [0, 1, 2], normalize=True)
    plt.show()
    np.savetxt('ideal_train/error{}.txt'.format(idx), np.asarray(wrongidx))


def ideal_test_rev(idx, thresh=None):
    wrong = 0
    wrongidx = []
    with open('ideal_train/{}.pickle'.format('gen_new_CW_vs_CCW'), 'rb') as f:
        direction_svm = pickle.load(f)
    with open('ideal_train/{}.pickle'.format('f2'), 'rb') as f:
        rot_non_svm = pickle.load(f)
    video_ves = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt'.format(idx))
    exp_label = np.loadtxt('const_index/lab/{}.txt'.format(idx))
    y_true = []
    y_pred = []
    for i in range(video_ves.shape[0]):
        exp = exp_label[i][2]
        if exp == 0:
            exp = 1
        elif exp == 1:
            exp = 0

        p, n, z, row = handle_row_rev(video_ves[i])
        if thresh is None:
            row_for_pred = np.asarray([p, n, z]).reshape((1, -1))

            rot_non_pred = rot_non_svm.predict(row_for_pred)

            if rot_non_pred == 1:
                pred = 2
            else:
                pred = direction_svm.predict(row_for_pred)
                proba = direction_svm.predict_proba(row_for_pred)
                if statistics.stdev(proba[0]) <= 0.5:
                    pred = 2
        else:
            if p > n:
                if n == 0:
                    pred = 1
                else:
                    if p / n >= thresh:
                        pred = 1
                    else:
                        pred = 2
            elif n > p:
                if p == 0:
                    pred = 0
                else:
                    if n / p >= thresh:
                        pred = 0
                    else:
                        pred = 2
            else:
                pred = 2

        if exp != pred:
            wrongidx.append([i, int(pred), int(exp)])
            wrong += 1
            if thresh is None:
                # print('Cell_index {} EXP: {}, PRED: {}, PROBA: {}-{:.2f}, Counts: {}'.format(i, exp, pred, proba, statistics.stdev(proba[0]), (p, n, z)))
                pass
        y_pred.append(pred)
        y_true.append(exp)
    # print('ACC: {}, xOTHx {}'.format((video_ves.shape[0] - wrong) / video_ves.shape[0], wrong_oth))
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    # print(y_true)
    # print(y_pred)
    plot_confusion_matrix(y_true, y_pred, [0, 1, 2], normalize=True)
    plt.show()
    np.savetxt('ideal_train/rev_error{}.txt'.format(idx), np.asarray(wrongidx))


def visualize(X_train, y_train, idx):
    c1, c2, c3 = None, None, None
    pca = PCA(n_components=2).fit(X_train)
    pca_2d = pca.transform(X_train)
    import pylab as pl
    for i in range(0, pca_2d.shape[0]):
        if y_train[i] == 0:
            c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif y_train[i] == 1:
            c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif y_train[i] == 2:
            c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
    plt.legend([c1, c2, c3], ['CW', 'CCW', 'OTH'])
    plt.title('{}'.format(idx))
    # plt.show()
    plt.savefig('ideal_train/plots/{}.jpg'.format(idx))
    plt.clf()


def reverse_generator():
    vec = np.loadtxt('ideal_train/new_train.txt')
    label = np.loadtxt('ideal_train/new_label.txt')
    rev_vec = np.flip(vec, 1)
    rev_vec = np.multiply(rev_vec, -1)
    rev_lab = []
    for l in label:
        if l == 0:
            l = 1
        elif l == 1:
            l = 0
        rev_lab.append(l)
    rev_lab = np.asarray(rev_lab)
    vec = np.concatenate((vec, rev_vec))
    lab = np.concatenate((label, rev_lab))
    print(vec.shape, lab.shape)
    np.savetxt('ideal_train/comb_vec.txt', vec)
    np.savetxt('ideal_train/comb_lab.txt', lab)


def gen_data(thresh):
    all_ = []
    labels = []
    random.seed(30)
    for i in range(3000):
        smaller = random.randint(1, int(24 / (thresh + 1)))
        larger = random.randint(int(smaller*thresh), 24 - smaller)
        zero = 24 - smaller - larger
        print(smaller, larger, zero, smaller + larger + zero)
        all_.append([smaller, larger, zero])
        labels.append(0)
        all_.append([larger, smaller, zero])
        labels.append(1)
    all_ = np.asarray(all_)
    labels = np.asarray(labels)
    print(all_.shape, labels.shape)
    np.savetxt('ideal_train/gen_vec.txt', all_)
    np.savetxt('ideal_train/gen_lab.txt', labels)


if __name__ == '__main__':
    # svm_classifier(simplified_new_train, new_label, "simplified_no_seg_e2.pickle")
    # svm_classifier(rot_non_train, rot_non_label, "24_no_seg_e1.pickle")
    # svm_classifier(new_train, new_label, "24_no_seg_e2.pickle")
    #
    # svm_classifier(new_train, new_label, "c1")
    # svm_classifier(rot_non_train, rot_non_label, "c2")
    #
    # svm_classifier(simplified_x, rot_non_label, "d2")
    # svm_classifier(count_train, new_label, "d1")
    # svm_classifier(count_train, new_label)
    #
    # np.savetxt('data4242019/simplified_x.txt', simplified_x)
    #
    # print(simplified_x.shape)
    # svm_classifier(simplified_x, labels)

    # test_video(15, "e2.pickle", "e1.pickle", "f2.pickle", "f1.pickle")
    # test_video(16, "e2.pickle", "e1.pickle", "f2.pickle", "f1.pickle")
    # test_video(17, "e2.pickle", "e1.pickle", "f2.pickle", "f1.pickle")
    # test_video(18, "e2.pickle", "e1.pickle", "f2.pickle", "f1.pickle")
    # test_video(32, "e2.pickle", "e1.pickle", "f2.pickle", "f1.pickle")

    # trainer()

    # test_video(15, "c2.pickle", "c1.pickle", "d2.pickle", "d1.pickle")
    # test_video(16, "c2.pickle", "c1.pickle", "d2.pickle", "d1.pickle")
    # test_video(17, "c2.pickle", "c1.pickle", "d2.pickle", "d1.pickle")
    # test_video(18, "c2.pickle", "c1.pickle", "d2.pickle", "d1.pickle")
    # test_video(32, "c2.pickle", "c1.pickle", "d2.pickle", "d1.pickle")

    # reverse_label(15)
    # reverse_label(16)
    # reverse_label(17)
    # reverse_label(18)
    # reverse_label(32)

    # test_video("15_reversed", "g2.pickle", "g1.pickle", "h2.pickle", "h1.pickle")
    # test_video("16_reversed", "g2.pickle", "g1.pickle", "h2.pickle", "h1.pickle")
    # test_video("17_reversed", "g2.pickle", "g1.pickle", "h2.pickle", "h1.pickle")
    # test_video("18_reversed", "g2.pickle", "g1.pickle", "h2.pickle", "h1.pickle")
    # test_video("32_reversed", "g2.pickle", "g1.pickle", "h2.pickle", "h1.pickle")

    # test_video("15", "g2.pickle", "g1.pickle", "h2.pickle", "h1.pickle")
    # test_video("16", "g2.pickle", "g1.pickle", "h2.pickle", "h1.pickle")
    # test_video("17", "g2.pickle", "g1.pickle", "h2.pickle", "h1.pickle")
    # test_video("18", "g2.pickle", "g1.pickle", "h2.pickle", "h1.pickle")
    # test_video("32", "g2.pickle", "g1.pickle", "h2.pickle", "h1.pickle")

    # mark_error(15)
    # mark_error(16)
    # mark_error(17)
    # mark_error(18)
    # mark_error(32)

    # reverse_video(15)
    # reverse_video(16)
    # reverse_video(17)
    # reverse_video(18)
    # reverse_video(32)

    # compare_vec(15)
    # rev_ver(15)
    # overall_exp = []
    # overall_exp.append(total_cw)
    # overall_exp.append(total_ccw)
    # overall_exp.append(total_oth)
    # overall_pred = []
    # overall_pred.append(pred_total_cw)
    # overall_pred.append(pred_total_ccw)
    # overall_pred.append(pred_total_oth)
    # bins = np.arange(3)
    #
    # for i in range(0, 3):
    #     overall_exp.append(0)
    #     overall_pred.insert(0, 0)
    #
    # x = np.arange(6)
    # plt.bar(x, height=overall_exp, color=(0.2, 0.4, 0.6, 0.6))
    # plt.bar(x, height=overall_pred)
    # plt.xticks(x, ['CW', 'CCW', 'OTH', 'CW', 'CCW', 'OTH'])
    # plt.savefig('bias/bias_overall_whole.jpg')

    # cross_val()

    # overall_acc = []
    # for i in range(10):
    #     thresh_ = i
    #     c1, t1 = clean_data_testing(15, thresh_)
    #     c2, t2 = clean_data_testing(16, thresh_)
    #     c3, t3 = clean_data_testing(17, thresh_)
    #     c4, t4 = clean_data_testing(18, thresh_)
    #     c5, t5 = clean_data_testing(32, thresh_)
    #     c6, t6 = clean_data_testing_reverse("15_reversed", thresh_)
    #     c7, t7 = clean_data_testing_reverse("16_reversed", thresh_)
    #     c8, t8 = clean_data_testing_reverse("17_reversed", thresh_)
    #     c9, t9 = clean_data_testing_reverse("18_reversed", thresh_)
    #     c10, t10 = clean_data_testing_reverse("32_reversed", thresh_)
    #     print("---------------------------------------------------")
    #     # total_acc = (c1+c2+c3+c4+c5) / (t1+t2+t3+t4+t5)
    #     total_stdev = c1+c2+c3+c4+c5
    #     # overall_acc.append(total_acc)
    #     overall_acc.append(total_stdev)
    #     # print("Thresh: {} <-> Overall_acc: {}".format(thresh_, total_acc))
    #     print("Thresh: {} <-> Overall_stdev: {}".format(thresh_, total_stdev))
    # # print("Overall_acc: {}".format(overall_acc))
    # print("Overall_stdev: {}".format(overall_acc))
    # # print("Best threshold is {}, with best acc {}".format(overall_acc.index(max(overall_acc)), max(overall_acc)))
    # print("Best threshold is {}, with best stdev {}".format(overall_acc.index(min(overall_acc)), min(overall_acc)))

    # rev_ver(15)

    # thresh_ = 3
    # for i in range(1, 33):
    #     if i < 10:
    #         idx = "0{}".format(i)
    #     else:
    #         idx = "{}".format(i)
    #     c1, t1 = clean_data_testing(idx, thresh_)


    # c6, t6 = clean_data_testing_reverse("15_reversed", thresh_)
    # c7, t7 = clean_data_testing_reverse("16_reversed", thresh_)
    # c8, t8 = clean_data_testing_reverse("17_reversed", thresh_)
    # c9, t9 = clean_data_testing_reverse("18_reversed", thresh_)
    # c10, t10 = clean_data_testing_reverse("32_reversed", thresh_)

    # counting_compare(32)

    # for i in range(2, 33):
    #     if i < 10:
    #         i = '0{}'.format(i)
    #     last_frame_patch(i)

    # sample100(15)
    # sample100(16)

    # feature_vector_filtering()
    # feature_vector_filtering_per_vid(15)
    # feature_vector_filtering_per_vid_by_ssim(15)


    # mark_non_ideal(idx, bad)

    # video_vertical_flip(15)

    # generate_flipped_loc(15)
    # flip_compare(15)
    # loc_check(15)
    # test()

    # frames_check()
    # patches_check()
    # for i in range(1, 10):
    #     if i < 10:
    #         i = '0{}'.format(i)
    #     data_loss_check(i)
    # data_loss_check('02')

    # vecs = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt'.format(15))
    # label = np.loadtxt("const_index/lab/{}.txt".format(15))
    # cell_id = 0
    # for vec in vecs:
    #     line = []
    #     for i in range(vec.shape[0]):
    #         if i % 2 == 0:
    #             line.append(vec[i])
    #     p = 0
    #     n = 0
    #     z = 0
    #
    #     for x in line:
    #         if x > 0:
    #             p += 1
    #         elif x < 0:
    #             n += 1
    #         elif x == 0:
    #             z += 1
    #     print('P {} N {} Z {}'.format(p, n, z))
    #     print(p/24, n/24, label[cell_id][2])
    #     print(line)
    #     print('---------------------------------------------------------')
    #     cell_id += 1

    # label1645()
    # idx, bad = plot_compare_filetrs('10', 0.9, 2)

    # gen_data(1.2)

    # filter_ideal_generate(1.5)

    # reverse_generator()

    # vec = np.loadtxt('ideal_train/gen_vec.txt')
    # label = np.loadtxt('ideal_train/gen_lab.txt')
    # svm_classifier(vec, label, 'gen_new_CW_vs_CCW')

    ideal_test('15', vote=True)
    ideal_test_rev('15')

    ideal_test('16', vote=True)
    ideal_test_rev('16')

    ideal_test('17', vote=True)
    ideal_test_rev('17')

    ideal_test('18', vote=True)
    ideal_test_rev('18')

#
    # mark_error('15')

    # for i in range(1, 32):
    #     if i < 10:
    #         i = '0{}'.format(i)
    #     ori_vec = np.loadtxt('/Users/cheng_stark/tmp/rotation_results/XY{}_video/vectors.txt'.format(i))
    #     ori_label = np.loadtxt('const_index/lab/{}.txt'.format(i))
    #     new_rows = []
    #     for row in ori_vec:
    #         p, n, z, new_row = handle_row(row)
    #         new_rows.append([p, n, z])
    #     new_rows = np.asarray(new_rows)
    #     # print(new_rows.shape)
    #     ori_label = ori_label[:, 2]
    #     # print(ori_label)
    #     visualize(new_rows, ori_label, i)
    #     print('Printed {}'.format(i))






