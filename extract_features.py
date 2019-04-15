# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:44:54 2018

@author: Yang Zhang

This version uses the a new method to get the histogram as:
    
    Amasses : Magnitude * Angle in each bin

"""
# %% Cell 1

import scipy
import numpy as np
import cv2
import math
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from sklearn import svm, datasets
import csv

from os import path

##

framenum=1
videonum=4


############################################################
#Choose the video to be analyzed, and set the parameter    #
############################################################
videof = path.join('data', '10x_XY0{}_video_8bit.avi'.format(videonum))
imagef = path.join('data', '10x_XY0{}_video_8bit.png'.format(videonum))
Resultf = path.join('data', 'labels0{}.txt'.format(videonum))
LabelFile = 'FijiAidXY0{}.csv'.format(videonum)

log_folder = './log'

#change patch size
LOCALPARA =2
#choose the index of cell to show histograms
TestInd=11
#magnitude threshold
MagThreshold=0.45
HistMagShown=[]
HistAngShown=[]
COMPLEXPARA=12
############################################################
#Start blob detection
############################################################
cap = cv2.VideoCapture(videof)
ret, frame = cap.read()
prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
StepWiseResult = []
NoRotationDetermin=[]

# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds 
params.minThreshold = 0
params.maxThreshold = 200
  
# Filter by Area.
params.filterByArea = True
params.maxArea = 2000
params.minArea = 200

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.03

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.7

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.3

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(frame)
im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

###########################################################################
# Show keypoints,display (uncomment those code to show the blobs           #
############################################################################
# cv2.imshow("Keypoints", im_with_keypoints)
# k = cv2.waitKey(0) & 0xFF
# if k == 27:         # wait for ESC key to exit
#     savepath='D:\mic3\20180622.png'
#     cv2.imwrite(savepath, im_with_keypoints)
#     cv2.destroyAllWindows()
# Detect Center and Size

#################################
# Define functions              #
#################################
def angle_between(v1, v2):
    ang1 = np.arctan2(*v1[::-1])
    ang2 = np.arctan2(*v2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    #angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    #return angle

def angle_between_array(y0, x0, y1, x1):
    ang1 = np.arctan2(y0, x0)
    ang2 = np.arctan2(y1, x1)
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def CoordinatesD(keypoints):
    # Detect Center and Size
    XCoord = []
    YCoord = []
    Size   = []
    for k in keypoints:
        XCoord.append(int(k.pt[0]))
        YCoord.append(int(k.pt[1]))
        Size.append(int(k.size))
    return XCoord, YCoord, Size

XCoord, YCoord, Size = CoordinatesD(keypoints)
CellNumb = len(XCoord)
print('{} cells in total'.format(CellNumb))

###########################################################################
# Displaying patch#  Uncomment to show  #
############################################################################
# 
# fig,ax = plt.subplots(1,figsize=(30,25))
# 
# # Display the image
# ax.imshow(prvs)
# 
# for p in range(CellNumb):
#     # Create a Rectangle patch, Add the patch to the Axes
#     left=(XCoord[p]-int(Size[p])/2)-2
#     bottom=(YCoord[p]-int(Size[p])/2)-2
#     lsize=Size[p]+4
#     rect = patches.Rectangle((left,bottom),lsize,lsize,linewidth=1,edgecolor='b',facecolor='none')
#     ax.add_patch(rect)
#     ax.plot(left,bottom,'ro')
#     ax.plot(XCoord[p],YCoord[p],'go')
#     ax.annotate(str(p), (left,bottom))
# plt.show()

#######################################################################
#Start analyzing all frames                  
#######################################################################
ret, frame = cap.read()
count=0

AllMag=[]
AllAng=[]

HoldNew=[]

FijiLable = []

num_bins = 24
bin_size = 360 / num_bins

num_rows = 0

with open(LabelFile, newline='') as csvfile:
    
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        if row:
            l = []
            #a=[x.strip() for x in row[0].split(',')]
            #if a[0] != '':
            if not row[0].isspace():
                for x in row[0].split(','):
                    a = x.strip()
                    if a == 'CCW':
                        l.append(-1)
                    elif a == 'NR':
                        l.append(0)
                    elif a == 'CW':
                        l.append(1)
                    else:
                        print('Warning: {} is not recognized as a label'.format(a))
                FijiLable.append(l)
FijiLable = np.asarray(FijiLable)              

idx_frame = 2
# %%Cell 2
while(ret):
    print('Working on frame {}'.format(idx_frame))
    idx_frame += 1

    nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Optical Flow detection
    #flow = cv2.calcOpticalFlowFarneback(prvs, nxt, 
    #           flow=None, pyr_scale=0.5, levels=3,
    #           winsize=5, iterations=20, poly_n=10,
    #           poly_sigma=1.1, flags=0)
    #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1],angleInDegrees=1)

    #Define the for optial flow direction detection #MethodRT1
       
    URCellRotationD = []
    
    NewHistLong=[]

    for i in range(CellNumb):
        #NewHist = [0] * num_bins
        
        #Get the coordinate of the center point and the four corner points of a window
        #in the whole image array.
        CenterX = XCoord[i]
        CenterY = YCoord[i]
        
        left = int(XCoord[i] - Size[i]/2 - LOCALPARA)
        top = int(YCoord[i] - Size[i]/2 - LOCALPARA)
        lsize = Size[i] + LOCALPARA * 2

        #Optical Flow detection
        prvs_patch = prvs[top:top+lsize, left:left+lsize]
        nxt_patch = nxt[top:top+lsize, left:left+lsize]
        flow = cv2.calcOpticalFlowFarneback(prvs_patch, nxt_patch, 
                   flow=None, pyr_scale=0.5, levels=3,
                   winsize=5, iterations=20, poly_n=10,
                   poly_sigma=1.1, flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1],
                                   angleInDegrees=1)

        ra = np.arange(lsize)
        ca = np.arange(lsize)
        xv, yv = np.meshgrid(ra, ca)
        xv += left
        yv += top
        yv = np.ravel(yv)
        xv = np.ravel(xv)

        #flow_y = np.ravel(flow[top:top+lsize, left:left+lsize, 0])
        #flow_x = np.ravel(flow[top:top+lsize, left:left+lsize, 1])
        flow_y = np.ravel(flow[:, :, 0])
        flow_x = np.ravel(flow[:, :, 1])
        relative_ang = angle_between_array(yv, xv, flow_y, flow_x)

        #mag_list = mag[top:top+lsize, left:left+lsize]

 #######################################################################
 #Display two histograms of the seletected cell 
 #######################################################################
        #hist1,bins1 = np.histogram(mag, bins=20, range=(0,5))                    
        #AllMag.append(hist1)
        #hist2,bins2 = np.histogram(relative_ang, bins=num_bins)
        #AllAng.append(hist2)
        #NewHistLong.append(NewHist)
    
    #Replace the prvs with the current frame
    ret, frame = cap.read() #Get the new frame

#Stepwise determination of the results  
    StepWiseResultArray = np.asarray(StepWiseResult)
    
    Determ = len(StepWiseResultArray)
    
    #   print ('cell # {}, NRFeature {}, CWFeature {} - CCWFeature {} = {}'.format(e,NRFeature,CWFeature,CCWFeature,CWFeature-CCWFeature))
    #Plot the results   
    #plt.figure(figsize=(20,10))
    #im = plt.imread(imagef)
    #implot = plt.imshow(im)
    
    # put a red dot, size 40, at 2 locations:
    #plt.scatter(XCoord, YCoord, c='r', s=10)
    MyAnswer = []
    #for m, txt in enumerate(RotatResult):
    #    plt.annotate(txt, (XCoord[m],YCoord[m]))
    #    MyAnswer.append([str(XCoord[m]),str(YCoord[m]),txt])
    
    #plt.show()
    
    AnswerList=[]
    AnswerListIndex=[]
    with open(Resultf, 'r') as f:
        x = f.read().split('\n')
    for i in range(len(x)):
        AnswerList.append(x[i].split('\t'))
        AnswerListIndex.append(x[i].split('\t')[:2])
           
    WRITE = []
    counter2=0
    for i in range(len(MyAnswer)):
        if MyAnswer[i][:2] in AnswerListIndex:
            #Display the flow here 
            AllAng[i] = [int(j) for j in AllAng[i]]
            AllMag[i] = [int(j) for j in AllMag[i]]
            AnswerIndex=AnswerListIndex.index(MyAnswer[i][:2])
            if AnswerList[AnswerIndex][2]!='Complex':
                counter2+=1
             #   Analyzer = FijiLable[...,count][]
                a = HoldNew[0][i]#+[Analyzer]
        
                WRITE.append(a)

    for w in range(len(WRITE)):
        WRITE[w].append(FijiLable[...,count][w])
    
    fn_results = 'datacollection0824_xy0{}.csv'.format(videonum)
    #with open(path.join(log_folder, fn_results), 'a') as csvFile:
    #    writer = csv.writer(csvFile)
    #    for i in range(len(WRITE)):
    #       writer.writerow(WRITE[i])
    #csvFile.close()
    
    
    prvs = nxt 
    AllAng=[]
    AllMag=[]
    HoldNew=[]
    count=count+1
cap.release()   