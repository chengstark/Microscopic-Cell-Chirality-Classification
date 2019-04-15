# import urllib.request
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import time
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
import sklearn
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
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from numpy import genfromtxt
# import simple_cnn as sic
# import imutils


def creat_pos_n_neg():
    for file_type in['neg']:

        for img in os.listdir(file_type):
            if file_type == 'neg':
                line = file_type + '/' + img + '\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)
            """
            elif file_type == 'pos':
                line = file_type + '/' + img + '1 0 0 50 50\n'
                with open('info.dat', 'a') as f:
                    f.write(line)
            """
    print('neg txt written!')


def overlap_check(x1, y1, w1, h1, x2, y2, w2, h2):
    if ((x1+w1) < x2) or ((x2+w2) < x1):  # one on the left of another
        return False
    if (y1 < (y2-h2)) or (y2 < (y1-h1)):  # one on the upper of another
        return False
    return True


def iou(x1, y1, w1, h1, x2, y2, w2, h2):
    area_inter = (min(x1+w1, x2+w2) - max(x1, x2)) * (min(y1+h1, y2+h2) - max(y1, y2))
    area_r1 = w1*h1
    area_r2 = w2*h2
    area_union = area_r1 + area_r2 - area_inter
    iou = area_inter / area_union * 100
    return iou


def cascade_detect(img_name):
    img = cv2.imread('{}.jpg'.format(img_name), 0)
    file = open('{}.txt'.format(img_name), 'a')

    print('1')
    # Here load the detector xml file, put it to your working directory
    # cell_cascade = cv2.CascadeClassifier('mydata/cascade.xml')
    cell_cascade = cv2.CascadeClassifier('cascade.xml')
    # cell_cascade = cv2.CascadeClassifier('cells-cascade-20stages.xml')
    print(cell_cascade)
    print('2')
    # Here used cell_cascade to detect cells, with size restriction the detection would be much faster
    cells = cell_cascade.detectMultiScale(img, minSize=(40, 40), maxSize=(55, 55))
    index1 = 0
    false_identified = set()
    for (x1, y1, w1, h1) in cells:
        # cv2.putText(img, str(index1),
        #             (x1 + 25, y1),
        #             cv2.FONT_HERSHEY_COMPLEX,
        #             0.8,
        #             color=(255, 0, 0))
        index2 = 0
        for (x2, y2, w2, h2) in cells:
            if x1 != x2 and y1 != y2:
                if overlap_check(x1, y1, w1, h1, x2, y2, w2, h2):
                    # cv2.putText(img, "##",
                    #             (x1 , y1),
                    #             cv2.FONT_HERSHEY_COMPLEX,
                    #             0.8,
                    #             color=(255, 0, 0))
                    iou_ = iou(x1, y1, w1, h1, x2, y2, w2, h2)
                    if iou_ > 15:
                        im1 = img[y1:y1 + h1, x1:x1 + w1]
                        (thresh1, im1) = cv2.threshold(im1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        im2 = img[y2:y2 + h2, x2:x2 + w2]
                        (thresh2, im2) = cv2.threshold(im2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        white1 = cv2.countNonZero(im1)
                        white2 = cv2.countNonZero(im2)
                        total1 = w1 * h2
                        total2 = w2 * h2
                        ratio1 = int(white1 * 100 / float(total1))
                        ratio2 = int(white2 * 100 / float(total2))
                        if ratio1 > ratio2:
                            false_identified.add(index2)
                            print("{}: {} -> {}/ {} -> {}/ {}".format(index1, index2, iou_, index2, ratio1, ratio2))
                        if ratio2 > ratio1:
                            false_identified.add(index1)
                            print("{}: {} -> {}/ {} -> {}/ {}".format(index1, index2, iou_, index1, ratio1, ratio2))
            index2 += 1
        index1 += 1

    delete_count = 0
    false_identified = list(false_identified)
    false_identified.sort()
    for i in false_identified:
        i = i - delete_count
        cells = np.delete(cells, i, axis=0)
        delete_count += 1

    print('3')
    # print(cells)
    # time.sleep(30)
    # Here we draw the result rectangles, (x, y) is the left-top corner coordinate of that triangle
    # We can just use (x, y) to locate each cell
    # w, h are the width and height
    for (x, y, w, h) in cells:
        # print((x,y,w,h))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        center_x = x + round(w / 2)
        center_y = y + round(h / 2)
        file.write('{} {} {} {}\n'.format(center_x, center_y, w, h))

    file.close()
    cv2.imwrite('result.jpg', img)


def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def get_rotation_template2(cell_index, frame_index, angle):
    cells_loc_ = cells_loc
    img = cv2.imread('data/06/frame{}.jpg'.format(frame_index), 0)
    x, y, w, h = cells_loc_[cell_index]
    img_cut = img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    rows, cols = img_cut.shape

    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(img_cut, mat, (cols, rows))
    cv2.imwrite("template_test/{}-{}.jpg".format(cell_index, frame_index), dst)
    return dst


def get_rotation_template(cell_index, frame_index, angle):
    # out_folder = 'cuts/{}'.format(index)
    # if not path.isdir(out_folder):
    #     os.makedirs(out_folder)

    img = cv2.imread('data/06/frame{}.jpg'.format(frame_index), 0)

    # print('cells type shape {}'.format(cells_loc.shape))
    # time.sleep(30)
    # print('1')
    row, col = img.shape
    center = tuple([cells_loc[cell_index][0], cells_loc[cell_index][1]])
    x, y, w, h = cells_loc[cell_index]
    # print(center)

    # In OpenCV, clockwise degree is negative, counter-clockwise degree is positive
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # print(rot_mat)
    result = cv2.warpAffine(img, rot_mat, (col, row))
    cv2.rectangle(result, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
    cut = result[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    # cv2.imwrite('cuts/{}/{}_{}.jpg'.format(index, index, angle), cut)
    return cut


def single_cell_direction3(cell_index, frame_index, video_degrees):
    # print(video_degrees)
    img_template = get_rotation_template(cell_index, frame_index+1, 0)
    degrees = np.arange(-degree_range, degree_range+1, 1)
    score_list = []
    for degree in degrees:
        img_compare = get_rotation_template(cell_index, frame_index, degree)
        # cv2.imwrite('compares/{}-{}-{}compare.jpg'.format(cell_index, frame_index, degree), img_compare)
        # cv2.imwrite('compares/{}-{}-{}template.jpg'.format(cell_index, frame_index, degree), img_template)
        # ssim_const = ssim(img_template, img_compare,
        #                  data_range=img_compare.max() - img_compare.min())
        ssim_const = find_ssim(img_template, img_compare, cell_index, frame_index, degree, video_degrees)
        # print('SSIM: {}'.format(ssim_const))
        # print('{}: ssim compare {:.4f}'.format(degree, ssim_const))\
        score_list.append(ssim_const)
        # print(ssim_const)

    max_value = max(score_list)
    print('BEST SSIM: {}'.format(max_value))
    max_index = score_list.index(max(score_list))
    best_degree = max_index - degree_range

    # print('Max value: {}'.format(max_value))
    # print('Max index: {}'.format(max_index))
    print('Cell {}/{} best degree: {}'.format(cell_index+1, cells_loc.shape[0], best_degree))

    # cv2.circle(img, (int(cells_loc[0][0]), int(cells_loc[0][1])), 5, (255, 0, 0), thickness=-1)
    # cv2.imshow('rotation', img)
    # cv2.waitKey(0)
    return best_degree


def single_cell_direction(cell_index, frame_index):
    img_template = get_rotation_template(cell_index, frame_index+1, 0)
    degrees = np.arange(-degree_range, degree_range+1, 1)
    score_list = []
    for degree in degrees:
        img_compare = get_rotation_template(cell_index, frame_index, degree)
        ssim_const = ssim(img_template, img_compare,
                          data_range=img_compare.max() - img_compare.min())
        # print('{}: ssim compare {:.4f}'.format(degree, ssim_const))
        score_list.append(ssim_const)

    max_value = max(score_list)
    max_index = score_list.index(max(score_list))
    best_degree = max_index - degree_range

    # print('Max value: {}'.format(max_value))
    # print('Max index: {}'.format(max_index))
    print('Cell {}/{} best degree: {}'.format(cell_index+1, cells_loc.shape[0], best_degree))

    # cv2.circle(img, (int(cells_loc[0][0]), int(cells_loc[0][1])), 5, (255, 0, 0), thickness=-1)
    # cv2.imshow('rotation', img)
    # cv2.waitKey(0)
    return best_degree

def single_cell_direction2(cell_index, frame_index, degree_range_, existed_degrees):
    img_template = get_rotation_template(cell_index, frame_index+1, 0)
    degrees_np = np.arange(-degree_range_, degree_range_+1, 1)
    degrees = degrees_np.tolist()
    n = len(degrees)
    d = find_bestfit(degrees, 0, n - 1, n, img_template, cell_index, frame_index)
    return d


def find_ssim(img_template_, img_compare_current, cell_index, frame_index, current_degree, existed_degrees):
    # cv2.imwrite("data/template_test/{}-{}-{}.jpg".format(frame_index, cell_index, current_degree), img_compare_current)
    # real index
    if frame_index <= 3:
        ssim_const_current = ssim(img_template_, img_compare_current, data_range=img_compare_current.max() - img_compare_current.min())
        return ssim_const_current

    if frame_index > 3:
        # print(existed_degrees)
        # print(frame_index-1)
        # print(existed_degrees)
        # print(len(existed_degrees[cell_index]))
        #cell_locs index
        previous_deg1 = existed_degrees[cell_index, frame_index-2]
        previous_deg2 = existed_degrees[cell_index, frame_index-3]
        previous_deg3 = existed_degrees[cell_index, frame_index-4]
        # picture index
        img_template1 = get_rotation_template(cell_index, frame_index-1, previous_deg1+current_degree)
        img_template2 = get_rotation_template(cell_index, frame_index-2, previous_deg2+previous_deg1+current_degree)
        img_template3 = get_rotation_template(cell_index, frame_index-3, previous_deg3+previous_deg2+previous_deg1+current_degree)

        # print("{}, {}, {}".format(previous_deg1+current_degree, previous_deg2+previous_deg1+current_degree, previous_deg3+previous_deg2+previous_deg1+current_degree))

        ssim1 = ssim(img_template_, img_template1, data_range=img_compare_current.max() - img_compare_current.min())
        ssim2 = ssim(img_template_, img_template2, data_range=img_compare_current.max() - img_compare_current.min())
        ssim3 = ssim(img_template_, img_template3, data_range=img_compare_current.max() - img_compare_current.min())
        current_ssim = ssim1*0.5 + ssim2*0.3 + ssim3*0.2
        print("{}: {}: {} -> {}".format(ssim1, ssim2, ssim3, current_ssim))


        return current_ssim


def find_bestfit(arr, low, high, n, img_template_, cell_index_, frame_index_):
    mid = int(low + (high - low) / 2)
    if mid == n - 1 or mid == 0:
        print('Cell {}/{} best degree: {}'.format(cell_index_ + 1, cells_loc.shape[0], arr[mid]))
        return arr[mid]
    degree_current = arr[mid]
    img_compare_current = get_rotation_template(cell_index_, frame_index_, degree_current)
    ssim_const_current = ssim(img_template_, img_compare_current, data_range=img_compare_current.max() - img_compare_current.min())
    degree_last = arr[mid-1]
    img_compare_last = get_rotation_template(cell_index_, frame_index_, degree_last)
    ssim_const_last = ssim(img_template_, img_compare_last, data_range=img_compare_last.max() - img_compare_last.min())
    # print("{}, {}, {}, {}, {}".format(mid+1, ssim_const_last <= ssim_const_current, mid == n - 1, mid, n))
    # print(mid == n-1)
    degree_next = arr[mid+1]
    img_compare_next = get_rotation_template(cell_index_, frame_index_, degree_next)
    ssim_const_next = ssim(img_template_, img_compare_next, data_range=img_compare_next.max() - img_compare_next.min())
    # ssim_const_last = round(ssim_const_last, 3)
    # ssim_const_next = round(ssim_const_next, 3)
    # ssim_const_current = round(ssim_const_current, 3)
    # print('Mid: {}'.format(mid))
    # print('{}, {}, {}'.format(ssim_const_last, ssim_const_current, ssim_const_next))
    if ssim_const_current < ssim_const_next and ssim_const_current < ssim_const_last:
        if ssim_const_next >= ssim_const_last:
            # print("Moved to next, deerrir")
            return find_bestfit(arr, (mid + 1), high, n, img_template_, cell_index_, frame_index_)
        else:
            # print("Moved to last, deerror ")
            return find_bestfit(arr, low, (mid - 1), n, img_template_, cell_index_, frame_index_)
    if(ssim_const_next <= ssim_const_current and ssim_const_last <= ssim_const_current) or 0:
        # print('SSIM: {}'.format(ssim_const_current))
        print('Cell {}/{} best degree: {}'.format(cell_index_ + 1, cells_loc.shape[0], arr[mid]))
        return arr[mid]
    elif mid > 0 and ssim_const_last > ssim_const_current:
        # print("Moved to last")
        return find_bestfit(arr, low, (mid - 1), n, img_template_, cell_index_, frame_index_)
    else:
        # print("Moved to nexr")
        return find_bestfit(arr, (mid + 1), high, n, img_template_, cell_index_, frame_index_)


def single_frame_detection(cells_loc_, frame_index, video_degrees):
    cells_degrees = np.zeros((cells_loc_.shape[0], 1))
    # print(cells_degrees)
    for i in range(0, cells_loc_.shape[0]):
        # degree = single_cell_direction2(i, frame_index, 10, video_degrees)
        # degree = single_cell_direction3(i, frame_index, video_degrees)
        degree = single_cell_direction(i, frame_index)
        cells_degrees[i][0] = degree
    # np.savetxt('frame1_rotations.txt', cells_degrees)
    # print('frame {} detection completed\n'.format(frame_index))
    # cells_degrees = np.reshape(cells_degrees, ())
    # print('cells_degree shape {}'.format(cells_degrees.shape))
    return cells_degrees


def draw_histograms(video_degrees, mode):
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
        plt.savefig('data/01_histograms/{}Cell{}.jpg'.format(mode, i))
        print('{}/{}: histogram written'.format(i+1, video_degrees.shape[0]))
    print('all histograms have been written')


def draw_kmeans(img_,results_,imgname, cells_loc_, foldername):
    # print('video_degrees shape {}'.format(video_degrees_.shape))
    # kmeans = KMeans(n_clusters=cluster, random_state=0).fit(video_degrees)
    result_labels = results_
    # print('labels {}'.format(result_labels))
    # print('labels shape {}'.format(len(result_labels)))
    # print(len(cells_loc_))

    for i in range(0, cells_loc_.shape[0]):
        if result_labels[i] == 1:#green for CCW
            color = (255, 0, 0)
        elif result_labels[i] == 2:#Blue for Complex/Other
            color = (0, 255, 0)
        elif result_labels[i] == 0:#Red for CW BGR RGB
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)
        h = cells_loc[i][2]
        cv2.rectangle(img_, (int(cells_loc_[i][0]-h/2), int(cells_loc_[i][1]-h/2)),(int(cells_loc_[i][0]+h/2), int(cells_loc_[i][1]+h/2)), color, 2)

        cv2.putText(img_, '{}'.format(i),
                    (int(cells_loc_[i][0])-25, int(cells_loc_[i][1])-25),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    color=color)

    cv2.imwrite('{}/{}'.format(foldername, imgname), img_)
    print('{} wirtten'.format(imgname))
    # cv2.imshow('rotation', img)
    # cv2.waitKey(0)



def draw_frames(video_degrees):
    for frame in range(0, video_degrees.shape[1]):
        for i in range(0, video_degrees.shape[0]):
            if video_degrees[i][frame] > 0:
                color = (0, 255, 0)
            elif video_degrees[i][frame] < 0:
                color = (255, 0, 0)
            else:
                continue

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


def match_labels_pervid(labeltxt, vectorstxt, rotationfile,error_margin_):
    print("Start matching...")
    labeledvectors_ = []
    with open(labeltxt, 'rb') as f:
        labels = np.loadtxt(f)
    with open(vectorstxt, 'rb') as f1:
        vectors = np.loadtxt(f1)
    with open(rotationfile, 'rb') as f2:
        vectors24 = np.loadtxt(f2)
    error_margin = error_margin_
    for i in range(0, len(vectors)):
        for j in range(0, len(labels)):
            vv = vectors[i]
            vl = labels[j]
            if (vv[0]<=vl[0]+error_margin and vv[0]>=vl[0]-error_margin) \
                    and (vv[1]<=vl[1]+error_margin and vv[1]>=vl[1]-error_margin):
                tmp = [i,vl[2]]
                labeledvectors_.append(tmp)

    final = []
    for i in range (0, len(labeledvectors_)):
        vector24tmp = vectors24[i].tolist()
        vector24tmp.append(labeledvectors_[i][1])
        final.append(vector24tmp)
    # labeledvectors = np.asarray(final)
    # print(labeledvectors)
    print(len(final))

    '''
    np.savetxt(outputfile,labeledvectors)
    print("Matching finished, labeled vectors saved")
    '''
    return final


def extract_frames_from_vid(videofilename, folder_index):
    cap = cv2.VideoCapture(videofilename)
    path = os.getcwd()
    print(path)
    videonum = 1
    video_capture = cv2.VideoCapture(videofilename)
    cap.open(videofilename)
    i = 1
    index = 1
    print(video_capture.isOpened())
    length = count_frames_manual(cap)
    print(length)
    perlength = length / 25
    print(perlength)
    while (i <= length):
        ret, frame = video_capture.read()
        if int(i % perlength) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Extracting frame{}".format(index))
            cv2.imshow('frame', gray)
            cv2.imwrite('data/0{}/frame{}.jpg'.format(folder_index,index), gray)
            print('frame{} written'.format(index))
            index += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        i += 1

    video_capture.release()
    cv2.destroyAllWindows()
    print('done')
    time.sleep(30)


def identify_rotation(original, train_source):
    # read in raw data and labelled training data
    celldata = genfromtxt(train_source, delimiter=',')
    with open(original, 'rb') as f1:
        original = np.loadtxt(f1)
    # shape
    # celldata.shape
    # original.shape

    # test/train set splitting preparation
    OX = original
    y_ = celldata[:,24]
    X_ = np.delete(celldata, 24, 1)
    X = np.asarray(X_)
    y = np.asarray(y_)
    # test/train set splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    # model training
    clf = svm.SVC(gamma=0.001,C=100, random_state=50)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    # result statistics
    print("Model precision analysis: \n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # save model
    filename = 'model.pickle'
    pickle.dump(clf, open(filename, 'wb'))
    print("Model Saved\n")
    # predict raw data
    result = clf.predict(OX)
    print("Result: ")
    print(result)
    np.savetxt('result(model_genrated).txt', result, delimiter=", ", fmt="%s")
    print("\nResult saved to result(model_genrated).txt\n")
    return result


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
    print(test_labels)
    print(test_result)
    train_acc = (train_labels == train_result).sum() / float(train_labels.size)
    test_acc = (test_labels == test_result).sum() / float(test_labels.size)
    print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('test_score {}'.format(test_score))
    # normalized_score = array_normalize(test_score)
    # print(normalized_score[0:10])
    # normalized_score = np.reshape(normalized_score, (normalized_score.shape[0], 1))
    # print('normalized shape {}'.format(normalized_score.shape))
    # print('normalized {}'.format(normalized_score[195:205]))
    # print('result {}'.format(test_result[195:205]))
    # print('test_score shape {}'.format(test_score.shape))
    return train_acc, test_acc, my_svm
    # return normalized_score
    # return test_score
    # time.sleep(30)


def SVM_voting(raw_set, model_infos):
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

        # print(voting_seq)
        for model in sorted_models:
            pred = model.predict(raw_set)
            for i in range(0, model_index):
                voting_seq.append(pred[cell_index])
            model_index += 1
        # print(voting_seq)
        most_common, num_most_common = Counter(voting_seq).most_common(1)[0]
        final_cell_rotate = most_common
        final_result.append(int(final_cell_rotate))
    print(final_result)
    return final_result


def loadModelandPredict(modelfilename, predset):
    loaded_model = pickle.load(open(modelfilename, 'rb'))
    with open(predset, 'rb') as f1:
        pred = np.loadtxt(f1)
    result = loaded_model.predict(pred)
    print(result)
    return result


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
    print(test_labels)
    print(test_result)
    train_acc = (train_labels == train_result).sum() / float(train_labels.size)
    test_acc = (test_labels == test_result).sum() / float(test_labels.size)
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('test_score {}'.format(test_score))
    # normalized_score = array_normalize(test_score)
    # print(normalized_score[0:10])
    # normalized_score = np.reshape(normalized_score, (normalized_score.shape[0], 1))
    # print('normalized shape {}'.format(normalized_score.shape))
    # print('normalized {}'.format(normalized_score[195:205]))
    # print('result {}'.format(test_result[195:205]))
    # print('test_score shape {}'.format(test_score.shape))
    return train_acc, test_acc, my_svm
    # return normalized_score
    # return test_score
    # time.sleep(30)


def identify_rotation2 (vectors, labels, raw):
    # read in raw data and labelled training data
    with open(vectors, 'rb') as f0:
        v = np.loadtxt(f0)
    with open(labels, 'rb') as f1:
        l = np.loadtxt(f1)
    # shape
    with open(raw, 'rb') as f1:
        original = np.loadtxt(f1)
    # shape
    # celldata.shape
    # original.shape

    # test/train set splitting preparation
    OX = original
    # test/train set splitting preparation
    y_ = l
    X_ = v
    X = np.asarray(X_)
    y = np.asarray(y_)
    # test/train set splitting

    clf = svm.SVC(gamma=0.001, C=100000, random_state=50)
    # clf = svm.SVC(kernel = 'linear', C = 1)

    for i in range(0,100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        # model training
        clf.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # result statistics
    print("Model precision analysis: \n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # save model
    filename = 'model_resaved.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)
        print('Model Saved!')
    pickle.dump(clf, open(filename, 'wb'))
    print("Model Saved\n")
    # predict raw data
    result = clf.predict(OX)
    print("Result: ")
    print(result)
    np.savetxt('result(model_genrated).txt', result, delimiter=", ", fmt="%s")
    print("\nResult saved to result(model_genrated).txt\n")
    return result


def accuracy_test (vectors, labels, modelfilename):
    # read in raw data and labelled training data
    with open(vectors, 'rb') as f0:
        v = np.loadtxt(f0)
    with open(labels, 'rb') as f1:
        l = np.loadtxt(f1)
    # shape
    model = pickle.load(open(modelfilename, 'rb'))
    # shape
    # celldata.shape
    # original.shape

    # test/train set splitting preparation
    # test/train set splitting preparation
    y_ = l
    X_ = v
    X = np.asarray(X_)
    y = np.asarray(y_)
    # test/train set splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # model training
    y_pred = model.predict(X_test)
    # result statistics
    print("Model precision analysis: \n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # save model

def count_frames_manual(video):
    total = 0
    while True:
        (grabbed, frame) = video.read()
        if not grabbed:
            break
        total += 1
    return total


def make_video(image_folder_, video_name_, fourcc_, fps):
    image_folder = image_folder_
    video_name = video_name_
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*fourcc_)  # 'x264' doesn't work
    video = cv2.VideoWriter(video_name, fourcc,fps , (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()


def center_tracking(cells_loc_):
    # tracker = cv2.TrackerBoosting_create()
    for i in range(0, cells_loc_.shape[0]):
        tracker = cv2.TrackerKCF_create()
        h = cells_loc_[i][2]
        frame = cv2.imread('data/06/frame1.jpg')
        bbox = (int(cells_loc_[i][0]-h/2), int(cells_loc_[i][1]-h/2), int(cells_loc_[i][0]+h/2), int(cells_loc_[i][1]+h/2))
        print(bbox[0], bbox[1], bbox[2], bbox[3])
        tracker.init(frame, bbox)
        for j in range(1, 25):
            file = open('tracking/tracking_frame{}.txt'.format(j), 'a')
            frame = cv2.imread('data/06/frame{}.jpg'.format(j))
            # print(frame)
            ok, bbox = tracker.update(frame)
            # print(bbox[0], bbox[1], bbox[2], bbox[3])
            if ok:
                print('check')
                file.write('{} {} {} {}\n'.format(bbox[0], bbox[1], h, h))
            else:
                print('fail')
                file.write('{} {} {} {}\n'.format(int(cells_loc_[i][0]-h/2), int(cells_loc_[i][1]-h/2), h, h))


def movingaverage(degrees, n):
    # smoothed = np.empty([len(degrees), len(degrees[0])])
    smoothed_degrees= []
    weights = np.exp(np.linspace(-1., 0., n))
    weights /= weights.sum()
    for eachcell in degrees:
        smoothed_row = np.convolve(eachcell, weights, mode='full')[:len(eachcell)]
        smoothed_degrees.append(smoothed_row)
        print(len(smoothed_row))
    return np.array(smoothed_degrees)  # as a numpy array

if __name__ =='__main__':
    start_time = time.time()
    # extract_frames_from_vid("data/XY15_video.avi", 6)

    # #identify_rotation2('gt_vectors.txt', 'gt_labels.txt', )
    #
    # l0 = match_labels_pervid('data/labels01.txt','singleframe.txt', 'video_rotations.txt',15)
    # l1 = match_labels_pervid('data/labels02.txt', 'secondsingleframe.txt',  'second_video_rotations.txt',1.5)
    # l2 = match_labels_pervid('data/labels03.txt', 'thirdsingleframe.txt', 'third_video_rotations.txt',15)
    # l3 = match_labels_pervid('data/labels04.txt', 'forthsingleframe.txt', 'forth_video_rotations.txt',15)


    # sic.load_labels(1)
    # print('labels loaded')
    # time.sleep(30)

    # '''SVM_VOTE'''
    # raw_vectors = np.loadtxt('weighted_single_video_rotations.txt')
    # # print(raw_vectors)
    # tracc1, teacc1, model1 = svm_classifier()
    # tracc2, teacc2, model2 = svm_classifier2()
    # # pred1 = model1.predict_proba(raw_vectors)
    # # pred2 = model2.predict_proba(raw_vectors)
    # # print(pred1)
    # # print(pred2)
    # a = [(tracc1, teacc1, model1), (tracc2, teacc2, model2)]
    # print(model1.predict(raw_vectors))
    # print(model2.predict(raw_vectors))
    # result = SVM_voting(raw_vectors, a)
    # print(result)
    # print(model1.predict(raw_vectors))
    # print(model2.predict(raw_vectors))
    ''''''

    # cluster = 3
    # degree_range = 10
    img_name = 'forthsingleframe'
    img = cv2.imread('forthsingleframe.jpg', 1)
    cascade_detect(img_name)
    # cells_loc = np.loadtxt('{}.txt'.format(img_name))
    # # center_tracking(cells_loc)
    # # single_cell_direction(0, 0)
    # # time.sleep(30)
    # #
    # video_degrees = None
    # video_degrees = single_frame_detection(cells_loc, 1, video_degrees)
    #
    # print('inter frame 1/24 completed')
    # for i in range(2, 25):
    #     frame_degrees = single_frame_detection(cells_loc, i, video_degrees)
    #     video_degrees = np.concatenate((video_degrees, frame_degrees), axis=1)
    #     print('inter frame {}/24 completed'.format(i))
    # np.savetxt('test_single_video_rotations.txt', video_degrees)
    # video_degrees = np.loadtxt('sixth_video_rotations.txt')
    #
    # # results = identify_rotation2('gt_vectors.txt', 'gt_labels.txt', 'sixth_video_rotations.txt')
    #
    # # results = identify_rotation('fifth_vectors.txt', 'all_rotate.csv')
    #
    # # draw_histograms(video_degrees)
    #
    # # print("Loading models ... \n")
    # # loadModelandPredict("vectors.txt", "model.pickle")
    # '''
    # result_folder_name = '06_results'
    # for i in range(1, 25):
    #     image_name = "data/01/frame{}.jpg".format(i)
    #     one_img = cv2.imread(image_name)
    #     image_name = "06_result_frame{}.jpg".format(i)
    #     draw_kmeans(one_img, results, image_name, cells_loc, result_folder_name)
    # '''
    # # make_video('06_results', 'XY15_video_results.avi', 'MJPG', 6)
    #
    # # Release everything if job is finished
    # # loadModelandPredict('my_svm.pickle', 'sixth_video_rotations——222.txt')
    # # loadModelandPredict('my_svm.pickle', 'sixth_video_rotations.txt')
    # # accuracy_test('gt_vectors.txt', 'gt_labels.txt', 'my_svm.pickle')
    # # draw_frames(video_degrees)
    # print('Time of execution: {}'.format(start_time-time.time()))




