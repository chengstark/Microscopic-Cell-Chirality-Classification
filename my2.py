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
step = 1
vector_dimension = 24
global_degree_record = np.zeros([0, 0])
smoothing_stdv_thresh = 3.5 # stdev value
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
    print("falses------------------------------------------------")
    print(false_identified)
    # false box removal
    for i in false_identified:
        i = i - delete_count
        cells = np.delete(cells, i, axis=0)
        delete_count += 1



    print('3')
    indices_ = np.lexsort((cells[:, 0], cells[:, 1]))
    cells = np.array(cells)[indices_]
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


def new_method(img_, frame_index, cell_index):
    cpy = img_.copy()
    w, h = img_.shape
    img = img_.copy()
    img_c = img_.copy()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img_t = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    # img_t = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # cv2.imwrite('recenter_c/tmp_fixing/tmp{}_{}.jpg'.format(frame_index, cell_index), img_t)
    _, cnts, h_ = cv2.findContours(img_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    the_cnt = None
    dis_from_center = 999
    height, width = img_t.shape
    cx_final = None
    cy_final = None
    cnt_i = 0
    for c in cnts:
        print("----------")
        cv2.drawContours(cpy, [c], -1, (0, 0, 255), 1)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # set values as what you need in the situation
            cx, cy = width / 2, height / 2
        shortest_x = None
        shortest_y = None
        shortest_dis = 9999
        max_dis = 0
        max_x = None
        max_y = None
        for [coord] in c:
            x = coord[0]
            y = coord[1]
            dis = math.sqrt((cx - x)**2 + (cy - y)**2)
            if dis <= shortest_dis:
                shortest_dis = dis
                shortest_x = x
                shortest_y = y
            if dis >= max_dis:
                max_dis = dis
                max_x = x
                max_y = y
        # math.floor((shortest_dis + max_dis) / 2)
        zerod = np.zeros((height, width))
        zerod[max_x][max_y] = 255
        circled = cv2.circle(img_t, (cx, cy), int(max_dis), (0, 0, 255), cv2.FILLED)
        kernel = np.ones((1, 1), np.uint8)
        circled = cv2.erode(circled, kernel, iterations=1)
        cv2.imwrite("new_method/full/im{}_cell{}.jpg".format(frame_index, cell_index), zerod)
        # print(cx)
        # print(cy)
        # print(x)
        # print(y)
        cnt_i += 1

def single_cell_direction(cell_index, frame_index, cells_loc):
    start = time.time()
    img_ = frames[frame_index]
    x, y, w, h = cells_loc[cell_index]
    img_ = frames[frame_index]
    # im = img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    # if angle == 0:
    #     cv2.imwrite('tmp2/tmp2{}{}.jpg'.format(cell_index, frame_index), im)
    #     return im
    img = img_[int(y - h/2):int(y + h/2), int(x - w/2):int(x + w/2)]
    new_method(img)



def single_frame_detection(cells_loc, frame_index):
    cells_degrees = np.zeros((cells_loc.shape[0], 1))
    for i in range(0, cells_loc.shape[0]):
        single_cell_direction(i, frame_index, cells_loc)

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
        global_degree_record = np.zeros([cells_num, vector_dimension])
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
            for cell_index in range(0, cells_loc.shape[0]):
                for frame_index in range(len(frames)):
                    frame = frames[frame_index]
                    x, y, w, h = cells_loc[cell_index]
                    img = frame[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                    new_method(img, frame_index, cell_index)


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

        time_per_image_average = sum(time_per_frame) / float(len(time_per_frame))


        print('Detected {} cells'.format(train_result.shape[0]))
        print('num_cw {}, num_ccw {}, num_other {}'.format(num_cw, num_ccw, num_other))
        print('ratio_cw {:.1f}%, ratio_ccw {:.1f}%, ratio_other {:.1f}%'.format(ratio_cw, ratio_ccw, ratio_other))
        print('Average time per frame is {:.4f}s'.format(time_per_image_average))
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


def plot_mse(score_list, cell_index, frame_index):
    plt.figure()
    r = np.arange(-degree_range, degree_range+1, step)
    r.flatten()
    r = r.tolist()
    plt.plot(r, score_list, marker='o')
    # r_ = np.array(r)
    # s_ = np.array(score_list)
    # z = np.polyfit(r_, s_, r_.shape[0]-1)
    z = score_list
    mean = (z[degree_range-1]+z[degree_range+1])/2
    if z[degree_range] > mean:
        z[degree_range] = mean
    plt.plot(r, z, color="red", marker='^')
    plt.xlabel('Degrees')
    plt.ylabel('mse')
    plt.title('frame {} Cell {}: Rotation Degrees'.format(frame_index, cell_index))
    plt.savefig('validations/mse/exp2/frame{}_cell{}.jpg'.format(frame_index, cell_index))


def plot_ssim(score_list, cell_index, frame_index):
    plt.figure()
    plt.figure()
    r = np.arange(-degree_range, degree_range + 1, step)
    r.flatten()
    r = r.tolist()
    plt.plot(r, score_list, marker='o')
    plt.xlabel('Degrees')
    plt.ylabel('ssim')
    plt.title('frame {} Cell {}: Rotation Degrees'.format(frame_index, cell_index))
    plt.savefig('validations/ssim/frame{}_cell{}.jpg'.format(frame_index, cell_index))


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
