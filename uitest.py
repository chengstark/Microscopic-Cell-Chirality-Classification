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

# from PyQt5 import QtWidgets, QtGui
import time
import cv2
import shutil



video_name = ''
img_name = 'frame0.jpg'
svm_file = 'my_svm.pickle'
# svm_file = 'model2.pickle'
tmp_folder = path.expanduser('~/tmp')
result_folder = path.expanduser('~/tmp/rotation_results')
source_folder = path.expanduser('~/videos')
video_folder = path.join(result_folder, video_name[0:-4])
txt_path = path.join(video_folder, '{}_loc.txt'.format(video_name[0:-4]))
img_path = path.join(video_folder, img_name)
vectors_path = path.join(video_folder, 'vectors.txt')
model_infos = []
cluster = 3
degree_range = 20
vector_dimension = 24
global_degree_record = np.zeros([0, 0])

num_cells = 0

frames = []
time_per_frame = []
rotation_data_recorder = []

# def prepare_frames(source_video):
def prepare_frames(source_video):
    # source_video = fname
    # source_video = path.join(source_folder, video_name)
    cap = cv2.VideoCapture(source_video)
    # cap.open(source_video)
    # cap = cv2.VideoCapture()

    # video_folder = path.join(result_folder, video_name[0:-4])
    print('video folder {}'.format(video_folder))
    print('video name is {}'.format(video_name))
    if not path.isdir(video_folder):
        os.makedirs(video_folder)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gap = math.floor(length / (vector_dimension + 1))
    print('length of the video: {} frames'.format(length))
    print('gap is {}'.format(gap))
    print('Is cap opened? {}'.format(cap.isOpened()))
    count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('frame', gray)
        if count % gap == 0:
            index = int(count / gap)
            frames.append(gray)
            filename = path.join(video_folder, 'frame{}.jpg'.format(index))
            cv2.imwrite(filename, gray)
            print('At {}, frame{}.jpg has been saved'.format(count, index))

        if count / gap == vector_dimension:
            break
        count = count + 1

    cap.release()

# determine whether two boxes overlap
def overlap_check(x1, y1, w1, h1, x2, y2, w2, h2):
    if ((x1+w1) < x2) or ((x2+w2) < x1):  # one on the left of another
        return False
    if (y1 < (y2-h2)) or (y2 < (y1-h1)):  # one on the upper of another
        return False
    return True

# iou calculation
def iou(x1, y1, w1, h1, x2, y2, w2, h2):
    area_inter = (min(x1+w1, x2+w2) - max(x1, x2)) * (min(y1+h1, y2+h2) - max(y1, y2))
    area_r1 = w1*h1
    area_r2 = w2*h2
    area_union = area_r1 + area_r2 - area_inter
    iou = area_inter / area_union * 100
    return iou

def cascade_detect(img):
    # img = cv2.imread('{}.jpg'.format(img_name), 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    file = open(txt_path, 'w')
    # file = open('singleframe4.txt', 'w')

    print('1')
    # Here load the detector xml file, put it to your working directory
    # cell_cascade = cv2.CascadeClassifier('mydata/cascade.xml')
    cell_cascade = cv2.CascadeClassifier('cascade.xml')
    # cell_cascade = cv2.CascadeClassifier('cells-cascade-20stages.xml')
    print(cell_cascade)
    print('2')
    # Here used cell_cascade to detect cells, with size restriction the detection would be much faster
    cells = cell_cascade.detectMultiScale(img, minSize=(50, 50), maxSize=(55, 55))
    # print(len(cells))
    # cells = cell_cascade.detectMultiScale(img, maxSize=(50, 50))

    # false positive identification removal
    # check for overlapping bounding boxes
    index1 = 0
    false_identified = set()
    for (x1, y1, w1, h1) in cells:
        index2 = 0
        for (x2, y2, w2, h2) in cells:
            if x1 != x2 and y1 != y2:
                if overlap_check(x1, y1, w1, h1, x2, y2, w2, h2):
                    print("{}, {}".format(index1, index2))
                    iou_ = iou(x1, y1, w1, h1, x2, y2, w2, h2)
                    # 15 is the threshold for IOU
                    print(iou_)
                    if iou_ > 15:
                        # using white pixel percentage to determine the correct box
                        im1 = img[y1:y1 + h1, x1:x1 + w1]
                        _, im1 = cv2.threshold(im1, 55, 255, cv2.THRESH_BINARY)
                        # (thresh1, im1) = cv2.threshold(im1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        im2 = img[y2:y2 + h2, x2:x2 + w2]
                        _, im2 = cv2.threshold(im2, 55, 255, cv2.THRESH_BINARY)
                        # (thresh2, im2) = cv2.threshold(im2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        white1 = cv2.countNonZero(im1)
                        white2 = cv2.countNonZero(im2)
                        total1 = w1 * h2
                        total2 = w2 * h2
                        ratio1 = int(white1 * 100 / float(total1))
                        ratio2 = int(white2 * 100 / float(total2))
                        # cv2.putText(img, str(index1),
                        #             (x1 + 70, y1),
                        #             cv2.FONT_HERSHEY_COMPLEX,
                        #             0.8,
                        #             color=(255, 0, 0))
                        # cv2.putText(img, str(index2),
                        #             (x2 + 70, y2),
                        #             cv2.FONT_HERSHEY_COMPLEX,
                        #             0.8,
                        #             color=(255, 0, 0))
                        print("{}: {} -> {}/ {} -> {}/ {}".format(index1, index2, iou_, index1, ratio1, ratio2))

                        if ratio1 > ratio2:
                            false_identified.add(index2)
                            print(index2)
                        # print("{}: {} -> {}/ {} -> {}/ {}".format(index1, index2, iou_, index2, ratio1, ratio2))
                        if ratio2 > ratio1:
                            print(index1)
                            false_identified.add(index1)
        index2 += 1
    index1 += 1

    delete_count = 0
    false_identified = list(false_identified)
    false_identified.sort()
    print(len(false_identified))
    # false box removal
    for i in false_identified:
        i = i - delete_count
        cells = np.delete(cells, i, axis=0)
        delete_count += 1



    print('3')
    print(cells)
    num_cells = len(cells)

    for i in range(0, num_cells):
        rotation_data_recorder.append([])
    # time.sleep(30)
    # Here we draw the result rectangles, (x, y) is the left-top corner coordinate of that triangle
    # We can just use (x, y) to locate each cell
    # w, h are the width and height
    i = 0
    for (x, y, w, h) in cells:
        cv2.circle(img, (int(x+w / 2), int(y+h / 2)), int(w / 2), (255, 0, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(img, '{}'.format(i),
                    (x, y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    color=(255, 255, 255))
        center_x = x + round(w / 2)
        center_y = y + round(h / 2)
        file.write('{} {} {} {}\n'.format(center_x, center_y, w, h))
        i = i+1

    file.close()
    detection_img = path.join(video_folder, 'detection.jpg')
    cv2.imwrite(detection_img, img)
    return i
    # cv2.imwrite('detection.jpg', img)

# def circle_cut(cell_index, frame_index, angle):
#     cells_loc = np.loadtxt(txt_path)
#     frame_path = path.join(video_folder, 'frame{}.jpg'.format(frame_index))
#     img = cv2.imread(frame_path, 0)
#     x, y, w, h = cells_loc[cell_index]
#     img_cut = img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
#     center = tuple(x, y)
#     cv2.circle(img_cut, center, w / 2, (255, 0, 0), -1)

def crop_circle(image, x, y, w, h):
    rows, cols = image.shape
    # cutting circle
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (int(x + w / 2), int(y + h / 2)), int(w / 2), (255, 255, 255), -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x1, y1, w1, h1 = cv2.boundingRect(contours[0])
    cropped = masked[y1:y1 + h1, x1:x1 + w1]
    return cropped

def get_rotation_template_direct_cut(cell_index, frame_index, angle):
    cells_loc = np.loadtxt(txt_path)
    # frame_path = path.join(video_folder, 'frame{}.jpg'.format(frame_index))
    # print("{}_{}".format(frame_path, angle))
    # img = cv2.imread(frame_path, 0)
    img = frames[frame_index]
    x, y, w, h = cells_loc[cell_index]
    # print("{}, {}".format(x, y))
    img_cut = img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    rows, cols = img_cut.shape
    cropped = crop_circle(img_cut, 0, 0, w, h)
    mat = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle, 1)
    rows1, cols1 = cropped.shape
    # print(mat)
    dst = cv2.warpAffine(cropped, mat, (cols1, rows1))
    _, dst = cv2.threshold(dst, 55, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("BW_apporach/{}_{}.jpg".format(cell_index, frame_index), thresh)
    return dst


def get_rotation_template(cell_index, frame_index, angle):
    cells_loc = np.loadtxt(txt_path)
    # frame_path = path.join(video_folder, 'frame{}.jpg'.format(frame_index))
    img = frames[frame_index]
    # img = cv2.imread(frame_path, 0)
    row, col = img.shape
    center = tuple([cells_loc[cell_index][0], cells_loc[cell_index][1]])
    x, y, w, h = cells_loc[cell_index]
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, (col, row))
    cv2.rectangle(result, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
    cut = result[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    cv2.imwrite('cut/{}_{}.jpg'.format(cell_index, frame_index, angle), cut)
    return cut


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def single_cell_direction(cell_index, frame_index):
    start = time.time()
    cells_loc = np.loadtxt(txt_path)
    img_template = get_rotation_template(cell_index, frame_index+1, 0)
    if frame_index >= 3:
        deg_a = rotation_data_recorder[cell_index][frame_index-1]
        deg_b = rotation_data_recorder[cell_index][frame_index-2]
        img_a = get_rotation_template(cell_index, frame_index-1, deg_a)
        img_b = get_rotation_template(cell_index, frame_index-2, deg_b+deg_a)
    degrees = np.arange(-degree_range, degree_range+1, 1)
    score_list = []
    score_list_a = []
    score_list_b = []
    # tmp = get_rotation_template_direct_cut(cell_index, frame_index+1, 0)

    # cv2.imwrite('comparison/{}_template_{}_{}.jpg'.format(cell_index, frame_index, 0), img_template)
    # print("{:.4f}".format(ssim(tmp, img_template, data_range=(tmp.max() - tmp.min()))))
    for degree in degrees:
        img_compare = get_rotation_template (cell_index, frame_index, degree)
        # cv2.imwrite('comparison/{}_compare_{}_{}.jpg'.format(cell_index, frame_index, degree), img_compare)

        # ssim_const = ssim(img_template, img_compare, data_range=img_compare.max() - img_compare.min())
        mse_index = mse(img_template, img_compare)
        if frame_index >= 3:
            mse_index_a = mse(img_a, img_compare)
            mse_index_b = mse(img_b, img_compare)
            score_list_a.append(mse_index_a)
            score_list_b.append(mse_index_b)
        # print(mse_index)
        # print('{}: ssim compare {:.4f}'.format(degree, ssim_const))

        # score_list.append(ssim_const)
        score_list.append(mse_index)

    max_value = min(score_list)
    min_index = score_list.index(min(score_list))
    best_degree = min_index - degree_range
    if frame_index >= 3:
        # print(score_list)
        # print(score_list_a)
        # print(score_list_b)
        min_index_a = score_list_a.index(min(score_list_a))
        best_degree_a = min_index_a - degree_range
        min_index_b = score_list_b.index(min(score_list_b))
        best_degree_b = min_index_b - degree_range
        if deg_a * deg_b > 0:
            print("Smoothing: {}, {}, {}".format(best_degree, best_degree_a, best_degree_b))
            best_degree = best_degree*0.7 + best_degree_a*0.2 + best_degree_b*0.1
    # print(score_list)
    # time.sleep(30)
    # print('Max value: {}'.format(max_value))
    # print('Max index: {}'.format(max_index))
    rotation_data_recorder[cell_index].append(best_degree)
    # print(rotation_data_recorder)
    print('Cell {}/{} best degree: {}'.format(cell_index+1, cells_loc.shape[0], best_degree))

    # cv2.circle(img, (int(cells_loc[0][0]), int(cells_loc[0][1])), 5, (255, 0, 0), thickness=-1)
    # cv2.imshow('rotation', img)
    # cv2.waitKey(0)

    end = time.time()
    time_per_frame.append(end - start)
    # print(end-start)
    return best_degree

def single_frame_detection(cells_loc, frame_index):
    cells_degrees = np.zeros((cells_loc.shape[0], 1))
    for i in range(0, cells_loc.shape[0]):
        degree = single_cell_direction(i, frame_index)
        cells_degrees[i][0] = degree
        # ex.text.setText('{}'.format(i))
    # np.savetxt('frame1_rotations.txt', cells_degrees)
    # print('frame {} detection completed\n'.format(frame_index))
    # cells_degrees = np.reshape(cells_degrees, ())
    # print('cells_degree shape {}'.format(cells_degrees.shape))
    return cells_degrees

def draw_histograms(video_degrees):
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
        plt.title('Cell {}: Rotation Degrees'.format(i))
        plt.savefig('data/01_histograms/Cell{}.jpg'.format(i))
        print('{}/{}: histogram written'.format(i+1, video_degrees.shape[0]))
    print('all histograms have been written')

def draw_kmeans(video_degrees):
    print('video_degrees shape {}'.format(video_degrees.shape))
    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(video_degrees)
    result_labels = kmeans.labels_
    print('labels {}'.format(result_labels))
    print('labels shape {}'.format(result_labels.shape))

    for i in range(0, cells_loc.shape[0]):
        if result_labels[i] == 1:
            color = (255, 0, 0)
        elif result_labels[i] == 0:
            color = (0, 255, 0)
        elif result_labels[i] == 2:
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        cv2.circle(img, (int(cells_loc[i][0]), int(cells_loc[i][1])), 15, color, thickness=-1)
        cv2.putText(img, '{}'.format(i),
                    (int(cells_loc[i][0])-25, int(cells_loc[i][1])-25),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    color=color)

    cv2.imwrite('video_kmeans.jpg', img)
    print('video_kmeans.jpg wirtten')
    # cv2.imshow('rotation', img)
    # cv2.waitKey(0)

    time.sleep(30)

def draw_frames(video_degrees):
    for frame in range(0, video_degrees.shape[1]):
        for i in range(0, video_degrees.shape[0]):
            if video_degrees[i][frame] > 0:
                color = (0, 255, 0)
            elif video_degrees[i][frame] < 0:
                color = (255, 0, 0)
            else:
                continue
            # frame_path = path.join()
            # img = cv2.imread()
            cv2.circle(img, (int(cells_loc[i][0]), int(cells_loc[i][1])), 15, color, thickness=-1)
            cv2.putText(img, '{}'.format(i),
                        (int(cells_loc[i][0]) - 25, int(cells_loc[i][1]) - 25),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        color=color)

        cv2.imwrite('data/01_rotations/frame{}_rotation.jpg'.format(frame), img)
        print('{}/{}: rotation.jpg written'.format(frame+1, video_degrees.shape[1]))
    print('all frames have been written')
    time.sleep(60)

    cv2.imshow('rotation', img)
    cv2.waitKey(0)

def process_data(vectors):
    # pca = PCA(n_components=3)
    # new_data = pca.fit_transform(vectors)
    avg = np.average(vectors, axis=1)
    avg = np.reshape(avg, (avg.shape[0], 1))
    std = np.std(vectors, axis=1)
    std = np.reshape(std, (std.shape[0], 1))
    sum = np.sum(vectors, axis=1)
    sum = np.reshape(sum, (sum.shape[0], 1))
    print('avg shape {}'.format(avg.shape))
    print('std shape {}'.format(std.shape))
    new_data = copy.copy(vectors)
    # new_data = np.concatenate((new_data, avg), axis=1)
    # new_data = np.concatenate((new_data, std), axis=1)
    # new_data = np.concatenate((new_data, sum), axis=1)
    # print('new_data.shape {}'.format(new_data.shape))
    return new_data

def svm_classifier():

    # filename = 'train/{}_svm.pickle'.format(class_name)
    gt_vectors = np.loadtxt('gt_vectors.txt')
    gt_labels = np.loadtxt('gt_labels.txt')
    gt_labels = np.reshape(gt_labels, (gt_labels.shape[0], 1))
    gt_vectors = process_data(gt_vectors)
    data = np.concatenate((gt_labels, gt_vectors), axis=1)
    np.random.shuffle(data)
    ratio = math.floor(data.shape[0] * 0.8)
    part_train = data[0:ratio]
    part_test = data[ratio:]
    # print('train{}, val{}'.format(part_train.shape, part_test.shape))

    train_data = part_train[:, 1:]
    train_labels = part_train[:, 0]
    test_data = part_test[:, 1:]
    test_labels = part_test[:, 0]

    # print('train data size {}, train label size {}'.format(train_data.shape, train_labels.shape))
    # print('test data size {}, test label size {}'.format(test_data.shape, test_labels.shape))

    train_data = sklearn.preprocessing.normalize(train_data, axis=1)
    # my_svm = sklearn.svm.LinearSVC(random_state=2, tol=1e-20, max_iter=1)
    my_svm = sklearn.svm.SVC(gamma=0.001, C=100000, random_state=50)

    # my_svm = svm.SVC(kernel='linear', C=0.0001)
    my_svm.fit(train_data, train_labels)
    # with open(filename, 'wb') as f:
    #     pickle.dump(my_svm, f)
    #     print('{} SVM model saved!'.format(class_name))
    # with open(filename, 'rb') as f:
    #     clf2 = pickle.load(f)
    train_result = my_svm.predict(train_data)
    test_result = my_svm.predict(test_data)
    # print('test_result shape {}'.format(test_result.shape))
    # print('test_labels shape {}'.format(test_labels.shape))
    # print(test_result)
    # test_score = my_svm.decision_function(test_data)
    # print('train_confidence {}'.format(test_score))
    # print('train_predicts {}'.format(test_result))
    # print(test_labels)
    # print(test_result)
    train_acc = (train_labels == train_result).sum() / float(train_labels.size)
    test_acc = (test_labels == test_result).sum() / float(test_labels.size)
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('test_score {}'.format(test_score))
    # normalized_score = array_normalize(test_score)
    # print(normalized_score[0:10])
    # normalized_score = np.reshape(normalized_score, (normalized_score.shape[0], 1))
    # print('normalized shape {}'.format(normalized_score.shape))
    # print('normalized {}'.format(normalized_score[195:205]))
    # print('result {}'.format(test_result[195:205]))
    # print('test_score shape {}'.format(test_score.shape))
    model_infos.append((train_acc, test_acc, my_svm))
    return train_acc, test_acc, my_svm
    # return normalized_score
    # return test_score
    # time.sleep(30)


def svm_classifier2():

    # filename = 'train/{}_svm.pickle'.format(class_name)
    gt_vectors = np.loadtxt('gt_vectors.txt')
    gt_labels = np.loadtxt('gt_labels.txt')
    gt_labels = np.reshape(gt_labels, (gt_labels.shape[0], 1))
    gt_vectors = process_data(gt_vectors)
    data = np.concatenate((gt_labels, gt_vectors), axis=1)
    np.random.shuffle(data)
    ratio = math.floor(data.shape[0] * 0.8)
    part_train = data[0:ratio]
    part_test = data[ratio:]
    # print('train{}, val{}'.format(part_train.shape, part_test.shape))

    train_data = part_train[:, 1:]
    train_labels = part_train[:, 0]
    test_data = part_test[:, 1:]
    test_labels = part_test[:, 0]

    # print('train data size {}, train label size {}'.format(train_data.shape, train_labels.shape))
    # print('test data size {}, test label size {}'.format(test_data.shape, test_labels.shape))

    train_data = sklearn.preprocessing.normalize(train_data, axis=1)
    # my_svm = sklearn.svm.LinearSVC(random_state=2, tol=1e-20, max_iter=1)
    my_svm = OneVsRestClassifier(sklearn.svm.LinearSVC(random_state=0))
    # my_svm = svm.SVC(kernel='linear', C=0.0001)
    my_svm.fit(train_data, train_labels)
    # with open(filename, 'wb') as f:
    #     pickle.dump(my_svm, f)
    #     print('{} SVM model saved!'.format(class_name))
    # with open(filename, 'rb') as f:
    #     clf2 = pickle.load(f)
    train_result = my_svm.predict(train_data)
    test_result = my_svm.predict(test_data)
    # print('test_result shape {}'.format(test_result.shape))
    # print('test_labels shape {}'.format(test_labels.shape))
    # print(test_result)
    # test_score = my_svm.decision_function(test_data)
    # print('train_confidence {}'.format(test_score))
    # print('train_predicts {}'.format(test_result))
    train_acc = (train_labels == train_result).sum() / float(train_labels.size)
    test_acc = (test_labels == test_result).sum() / float(test_labels.size)
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('test_score {}'.format(test_score))
    # normalized_score = array_normalize(test_score)
    # print(normalized_score[0:10])
    # normalized_score = np.reshape(normalized_score, (normalized_score.shape[0], 1))
    # print('normalized shape {}'.format(normalized_score.shape))
    # print('normalized {}'.format(normalized_score[195:205]))
    # print('result {}'.format(test_result[195:205]))
    # print('test_score shape {}'.format(test_score.shape))
    model_infos.append((train_acc, test_acc, my_svm))
    return train_acc, test_acc, my_svm
    # return normalized_score
    # return test_score
    # time.sleep(30)

def svm_classifier_all():

    # filename = 'train/{}_svm.pickle'.format(class_name)
    gt_vectors = np.loadtxt('gt_vectors.txt')
    gt_labels = np.loadtxt('gt_labels.txt')
    gt_labels = np.reshape(gt_labels, (gt_labels.shape[0], 1))
    # gt_vectors = process_data(gt_vectors)
    data = np.concatenate((gt_labels, gt_vectors), axis=1)

    np.random.shuffle(data)

    train_data = data[:, 1:]
    train_labels = data[:, 0]

    # print('train data size {}, train label size {}'.format(train_data.shape, train_labels.shape))
    # print('test data size {}, test label size {}'.format(test_data.shape, test_labels.shape))

    train_data = sklearn.preprocessing.normalize(train_data, axis=1)
    my_svm = sklearn.svm.LinearSVC(random_state=5, tol=1e-20, max_iter=5000)
    # my_svm = sklearn.svm.SVC(gamma=0.001, C=1000000, random_state=50)

    my_svm.fit(train_data, train_labels)
    # with open(filename, 'wb') as f:
    #     pickle.dump(my_svm, f)
    #     print('{} SVM model saved!'.format(class_name))
    # with open(filename, 'rb') as f:
    #     clf2 = pickle.load(f)
    train_result = my_svm.predict(train_data)
    # test_result = my_svm.predict(test_data)
    # print('test_result shape {}'.format(test_result.shape))
    # print('test_labels shape {}'.format(test_labels.shape))
    # print(test_result)
    # test_score = my_svm.decision_function(test_data)
    # print('train_confidence {}'.format(test_score))
    # print('train_predicts {}'.format(test_result))
    # print(test_labels)
    # print(test_result)
    train_acc = (train_labels == train_result).sum() / float(train_labels.size)
    test_acc = train_acc
    # test_acc = (test_labels == test_result).sum() / float(test_labels.size)
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('test_score {}'.format(test_score))
    # normalized_score = array_normalize(test_score)
    # print(normalized_score[0:10])
    # normalized_score = np.reshape(normalized_score, (normalized_score.shape[0], 1))
    # print('normalized shape {}'.format(normalized_score.shape))
    # print('normalized {}'.format(normalized_score[195:205]))
    # print('result {}'.format(test_result[195:205]))
    # print('test_score shape {}'.format(test_score.shape))
    model_infos.append((train_acc, test_acc, my_svm))
    return train_acc, test_acc, my_svm
    # return normalized_score
    # return test_score
    # time.sleep(30)

def svm_voting(raw_set, model_infos):
    noresults = len(model_infos)
    nocells = len(raw_set)
    # results = np.empty([nocells, noresults], dtype=int)
    acc_sorting = []
    index = 0
    for (tracc, teacc, model) in model_infos:
        # if abs(tracc - teacc) / tracc * 100 > 10:
        #     continue
        # else:
        acc_sorting.append([tracc*0.8 + teacc*0.2, index])
        index += 1
    acc_sorting.sort()
    sorted_models = []
    for acc, index in acc_sorting:
        sorted_models.append(model_infos[index][2])
    final_result = []
    for cell_index in range(0, nocells):
        model_index = 1
        voting_seq = []
        for model in sorted_models:
            pred = model.predict(raw_set)
            for i in range(0, model_index):
                voting_seq.append(pred[cell_index])
            model_index += 1
        most_common, num_most_common = Counter(voting_seq).most_common(1)[0]
        final_cell_rotate = most_common
        final_result.append(int(final_cell_rotate))

    return final_result

def cross_validation(n):
    train = []
    test = []
    models = []
    for i in range(0, n):
        train_acc, test_acc, my_svm = svm_classifier_all()
        # train_acc, test_acc, my_svm = svm_classifier()
        train.append(train_acc)
        test.append(test_acc)
        models.append(my_svm)
        print('{}/{}: train acc {:.4f}, test acc {:.4f}'.format(i + 1, n, train_acc, test_acc))

    best_index = test.index(max(test))
    best_svm = models[best_index]
    avg_train = sum(train) / len(train)
    avg_test = sum(test) / len(test)
    avg_test = sum(test) / len(test)
    print('avg_train {:.4f}, avg_test {:.4f}'.format(avg_train, avg_test))
    print('Best from {}: train acc {:.4f}, test acc {:.4f}'.format(best_index, train[best_index], test[best_index]))
    with open('my_svm.pickle', 'wb') as f:
        pickle.dump(best_svm, f)
        print('Best SVM model saved!'.format())

def get_frame(img, cells_loc, train_result, index):
    # cv2.imshow('shopw', img)
    # cv2.waitKey(0)
    for i in range(0, cells_loc.shape[0]):
        w, h = int(cells_loc[i][2]), int(cells_loc[i][3])
        x, y = int(cells_loc[i][0] - w//2), int(cells_loc[i][1] - h//2)
        # print('({},{}) with size ({},{})'.format(x, y, w, h))
        if train_result[i] == 0:    # Blue CW
            color = (255, 0, 0)
        elif train_result[i] == 1:  # Green CCW
            color = (0, 255, 0)
        else:                       # Red Complex
            color = (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, '{}'.format(i),
                    (x, y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    color=color)
    result_path = path.join(video_folder, 'result{}.jpg'.format(index))
    cv2.imwrite(result_path, img)
    return img



def get_video(video_folder):
    video_degrees = np.loadtxt(vectors_path)
    cells_loc = np.loadtxt(txt_path)
    # print('video_degrees size {}'.format(video_degrees.shape))
    clf2 = pickle.load(open(svm_file, 'rb'))
    # tracc1, teacc1, model1 = svm_classifier()
    # tracc2, teacc2, model2 = svm_classifier2()
    # print(tracc1, teacc1)
    # print(tracc2, teacc2)
    video_degrees = process_data(video_degrees)

    train_result = clf2.predict(video_degrees)
    # train_result = svm_voting(video_degrees, model_infos)
    # print('train result:\n{}'.format(train_result))
    # print('train result shape:\n{}'.format(train_result.shape))
    train_result = np.array(train_result)
    length = video_degrees.shape[1]
    # print('frame length {}'.format(length))
    img_path = path.join(video_folder, 'frame{}.jpg'.format(1))
    video_path = path.join(video_folder, video_name[0:-4] + '_result.avi')
    img = cv2.imread(img_path, 1)
    height, width = img.shape[0], img.shape[1]

    size = (width, height)
    fps = 6
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # video = cv2.VideoWriter(video_path, fourcc, fps, size)
    video = cv2.VideoWriter()
    video.open(video_path, fourcc, fps, size, True)
    for index in range(1, length+1):
        img_path = path.join(video_folder, 'frame{}.jpg'.format(index))
        img = cv2.imread(img_path, 1)
        result = get_frame(img, cells_loc, train_result, index)
        video.write(result)
        print('frame {} written!'.format(index))
    video.release()
    return train_result

def update_path(string_name):
    global video_name, video_folder, txt_path, img_path, vectors_path
    video_name = string_name
    count = 0
    video_name_ = video_name[0:-4]
    if os.path.exists(video_folder):
        # video_folder = path.join(video_folder, 'new')
        listOfFiles = os.listdir(result_folder)
        print("Video folder exists, created a duplicate folder")
        for file in listOfFiles:
            if video_name_ in file:
                count += 1
    print(count)
    if count != 0:
        video_name_ += '('
        video_name_ += str(count)
        video_name_ += ')'
    video_folder = path.join(result_folder, video_name_)
    txt_path = path.join(video_folder, '{}_loc.txt'.format(video_name_))
    img_path = path.join(video_folder, img_name)
    vectors_path = path.join(video_folder, 'vectors.txt')
    print('video name {}'.format(video_name))
    print('img_path {}'.format(img_path))
    print('video folder {}'.format(video_folder))

class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()
        self.mission_message = 'Welcome to cell rotation detection platform!'
        self.image_name = 'singleframe.jpg'
        self.imagePath = 'background.jpg'

    def initUI(self):

        loadVideoBtn = QPushButton("Load Video", self)
        # loadVideoBtn.move(80, 100)
        loadVideoBtn.setGeometry(50, 100, 140, 40)
        loadVideoBtn.clicked.connect(self.getfile)

        # frameButton = QPushButton("Prepare Frames", self)
        # frameButton.move(80, 100)
        # frameButton.setGeometry(100, 100, 130, 40)
        # frameButton.clicked.connect(self.frameClicked)

        detectButton = QPushButton("Detect Location", self)
        # detectButton.move(80, 200)
        detectButton.setGeometry(50, 200, 140, 40)
        detectButton.clicked.connect(self.locationClicked)

        rotationButton = QPushButton("Rotate Direction", self)
        # rotationButton.move(80, 300)
        rotationButton.setGeometry(50, 300, 140, 40)
        rotationButton.clicked.connect(self.rotationClicked)

        self.blue = QLabel(self)
        self.blue.setText('Blue for CW')
        self.blue.setGeometry(50, 350, 250, 250)
        newfont = QFont("Times", 16, QFont.Bold)
        self.blue.setFont(newfont)
        self.blue.setStyleSheet('color: blue')

        self.green = QLabel(self)
        self.green.setText('Green for CCW')
        self.green.setGeometry(50, 380, 250, 250)
        newfont = QFont("Times", 16, QFont.Bold)
        self.green.setFont(newfont)
        self.green.setStyleSheet('color: green')

        self.red = QLabel(self)
        self.red.setText('Red for NR/Cplx')
        self.red.setGeometry(50, 410, 250, 250)
        newfont = QFont("Times", 16, QFont.Bold)
        self.red.setFont(newfont)
        self.red.setStyleSheet('color: red')

        self.text = QLabel(self)
        self.text.setText('Waiting for results...')
        self.text.setGeometry(50, 600, 250, 250)
        newfont = QFont("Times", 16, QFont.Bold)
        self.text.setFont(newfont)

        self.pic = QLabel(self)
        self.pic.setPixmap(QPixmap('background2.jpg'))
        self.pic.setGeometry(300, 0, 1200, 900)
        # pic.resize(800, 400)
        self.pic.setScaledContents(True)

        self.pic.show()  # You were missing this.

        self.statusBar()
        self.resize(1500, 900)
        self.center()

        self.setWindowTitle('Cell Rotation Platform Demo')
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def getfile(self):
        self.statusBar().showMessage('Selecting Video...')
        fname = QFileDialog.getOpenFileName(self, 'Open file', source_folder)
        list = fname[0].split('/')
        load_name = list[-1]
        update_path(load_name)
        if not os.path.isfile(img_path):
            prepare_frames(fname[0])
        print('Frames have been prepared!')
        self.pic.setPixmap(QPixmap(img_path))
        self.statusBar().showMessage('You have selected {}'.format(load_name))
        self.text.setText('{} loaded!'.format(video_name))

    def frameClicked(self):
        self.statusBar().showMessage('Preparing frames for {}...'.format(video_name))
        if not os.path.isfile(img_path):
            prepare_frames(video_name)
        self.pic.setPixmap(QPixmap(img_path))
        self.statusBar().showMessage('24 frames from {} saved!'.format(video_name))

    def locationClicked(self):
        self.statusBar().showMessage('Cell location detection start...')
        detection_path = path.join(video_folder, 'detection.jpg')
        if not os.path.isfile(detection_path):
            img = cv2.imread(img_path, 1)
            cascade_detect(img)
        cells_num = np.loadtxt(txt_path).shape[0]
        global_degree_record = np.zeros([vector_dimension, cells_num])
        self.statusBar().showMessage('Cell location detection completed!')
        self.text.setText('{}:\n'
                          'Detected {} cells\n'.format(video_name, cells_num))
        self.pic.setPixmap(QPixmap(detection_path))

    def rotationClicked(self):
        self.statusBar().showMessage('Cell rotation detection start...')
        since = time.time()
        # if we have not calculated degree vectors, then do it
        if not os.path.isfile(vectors_path):
            cells_loc = np.loadtxt(txt_path)
            video_degrees = single_frame_detection(cells_loc, 1)
            print('inter frame 1/24 completed')
            self.text.setText('1/24 completed...')
            for i in range(1, 24):
                frame_degrees = single_frame_detection(cells_loc, i)
                video_degrees = np.concatenate((video_degrees, frame_degrees), axis=1)
                print('inter frame {}/24 completed'.format(i+1))
                self.text.setText('{}/24 completed...'.format(i+1))
            np.savetxt(vectors_path, video_degrees)
        # load the file and do classification
        train_result = get_video(video_folder)
        rotation_path = path.join(video_folder, 'result1.jpg')
        num_cw = np.count_nonzero(train_result == 0)
        num_ccw = np.count_nonzero(train_result == 1)
        num_other = np.count_nonzero(train_result == 2)
        ratio_cw = 100 * num_cw / train_result.shape[0]
        ratio_ccw = 100 * num_ccw / train_result.shape[0]
        ratio_other = 100 * num_other / train_result.shape[0]
        time_finished = time.time() - since

        time_per_image_average = sum(time_per_frame) / len(time_per_frame)


        print('Detected {} cells'.format(train_result.shape[0]))
        print('num_cw {}, num_ccw {}, num_other {}'.format(num_cw, num_ccw, num_other))
        print('ratio_cw {:.1f}%, ratio_ccw {:.1f}%, ratio_other {:.1f}%'.format(ratio_cw, ratio_ccw, ratio_other))
        print('Average time per frame is {:.0f}s'.format(time_per_image_average))
        print('Finished in {:.0f}m {:.0f}s'.format(time_finished // 60, time_finished % 60))

        self.pic.setPixmap(QPixmap(rotation_path))
        self.text.setText('{}:\n'
                          'Detected {} cells\n'
                          '{:.1f}% CW: {}\n'
                          '{:.1f}% CCW: {}\n'
                          '{:.1f}% NR/Cplx: {}\n'
                          'Finished in {:.0f}m {:.0f}s'.format(video_name,
                                                            train_result.shape[0],
                                                            ratio_cw, num_cw,
                                                            ratio_ccw, num_ccw,
                                                            ratio_other, num_other,
                                                            time_finished // 60, time_finished % 60))
        result_diary = path.join(video_folder, 'result_diary.txt')
        file = open(result_diary, 'w')
        file.write('{}:\n'
                          'Detected {} cells\n'
                          '{:.1f}% CW: {}\n'
                          '{:.1f}% CCW: {}\n'
                          '{:.1f}% NR/Cplx: {}\n'
                          'Finished in {:.0f}m {:.0f}s'.format(video_name,
                                                            train_result.shape[0],
                                                            ratio_cw, num_cw,
                                                            ratio_ccw, num_ccw,
                                                            ratio_other, num_other,
                                                            time_finished // 60, time_finished % 60))
        file.close()
        self.statusBar().showMessage('Cell rotation detection finished!')


def main_function():
    video_name = 'XY01_video.avi'
    # prepare_frames(video_name)
    # time.sleep(30)
    img_name = 'frame0.jpg'

    video_folder = path.join(result_folder, video_name[0:-4])
    txt_path = path.join(video_folder, '{}_loc.txt'.format(video_name[0:-4]))
    img_path = path.join(video_folder, img_name)
    vectors_path = path.join(video_folder, 'vectors.txt')

    img = cv2.imread(img_path, 1)
    cascade_detect(img)
    print('detection for {} completed!'.format(video_name))
    cells_loc = np.loadtxt(txt_path)

    video_degrees = single_frame_detection(cells_loc, 1)
    print('inter frame 1/24 completed')
    for i in range(1, 24):
        frame_degrees = single_frame_detection(cells_loc, i)
        video_degrees = np.concatenate((video_degrees, frame_degrees), axis=1)
        print('inter frame {}/24 completed'.format(i+1))
    np.savetxt(vectors_path, video_degrees)
    print('########## finished ##############')
    time.sleep(60)

    video_degrees = np.loadtxt('video_rotations.txt')
    ##########################################################################################

    video_degrees = np.loadtxt(vectors_path)
    print('video_degrees size {}'.format(video_degrees.shape))
    clf2 = pickle.load(open(svm_file, 'rb'))
    train_result = clf2.predict(video_degrees)
    print('train result:\n{}'.format(train_result))
    print('train result shape:\n{}'.format(train_result.shape))
    get_video(video_folder)

if __name__ == '__main__':

    # svm_classifier_all()
    # cross_validation(1000)
    # time.sleep(30)
    # img_name = 'frame0.jpg'
    #
    # video_folder = path.join(result_folder, video_name[0:-4])
    # txt_path = path.join(video_folder, '{}_loc.txt'.format(video_name[0:-4]))
    # img_path = path.join(video_folder, img_name)
    # vectors_path = path.join(video_folder, 'vectors.txt')
    #
    # with open('data/gt_vectors.txt', 'rb') as f1:
    #     vectors = np.loadtxt(f1)
    # # vectors = np.loadtxt('data/gt_vectors.txt')
    # new_data = process_data(vectors)
    # label = np.loadtxt('data/gt_labels.txt')
    # model = pickle.load(open('model2.pickle', 'rb'))
    # result = model.predict(vectors)
    # train_acc = (label == result).sum() / float(label.size)
    # print('predict {}'.format(result))
    # print('label {}'.format(label))
    # print('train acc {}'.format(train_acc))
    # time.sleep(30)

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
