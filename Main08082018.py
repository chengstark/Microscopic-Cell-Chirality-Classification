# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:44:54 2018

@author: Yang Zhang

This version uses the a new method to get the histogram as:
    
    Amasses : Magnitude depending on the angle degree

"""
# %% Cell 1

import scipy
import numpy as np
import cv2
import math
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import csv


#framenum=1
#videonum=4

for i in [1,2,3,4]:
    videonum= i
    ############################################################
    #Choose the video to be analyzed, and set the parameter    #
    ############################################################
    videof = r'D:\mic3\data\10x_XY0{}_video_8bit.avi'.format(videonum)
    imagef = r'D:\mic3\data\10x_XY0{}_video_8bit.tif'.format(videonum)
    Resultf = 'D:\mic3\data\labels0{}.txt'.format(videonum)
    LabelFile='D:\mic3\FijiAidXY0{}.csv'.format(videonum)
    #change patch size
    LOCALPARA =2
    #choose the index of cell to show histograms
    TestInd=11
    #magnitude threshold
    MagThrushold=0.45
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
    def angle_between(p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
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
    
    with open(LabelFile, newline='') as csvfile:
        
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            
            if row:
                a=[x.strip() for x in row[0].split(',')]
                if a[0]!='':
                    FijiLable.append([x.strip() for x in row[0].split(',')])
    FijiLable = np.asarray(FijiLable)              
    #%%Cell 2
    while(ret):
        
        nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Optical Flow detection
        flow = cv2.calcOpticalFlowFarneback(prvs,nxt, flow=None, pyr_scale=0.5,levels=3,winsize=5,iterations=20, poly_n=10,poly_sigma=1.1,flags=0)
    
        
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1],angleInDegrees=1)
    
        #Define the for optial flow direction detection #MethodRT1
           
        URCellRotationD = []
        NoRotation=[]  #1 stands for no rotation
        
        NewHistLong=[]
    
        for i in range(CellNumb):
    # =============================================================================
    #         Change bin size here
    # =============================================================================
            NewHist=[0]*10
            
            HistTest=[]
            HistTestAng=[]
            
            RelativeAng = []
            #Get the coordinate of the center point and the four corner points of a window
            #in the whole image array.
            CenterX=XCoord[i]
            CenterY=YCoord[i]
            
            left=int(XCoord[i]-Size[i]/2-LOCALPARA)
            bottom=int(YCoord[i]-Size[i]/2-LOCALPARA)
            lsize=Size[i]+LOCALPARA*2
    
            SingleNoRotation = 0
            for r in range(lsize):
                for c in range(lsize):
                    #The vector from the current point to 
                    GridR=left+r
                    GridC=bottom+c
                    RadiiLineVector = (CenterX-GridR,CenterY-GridC)
                    FlowVector = (flow[GridC][GridR][0],flow[GridC][GridR][1])
                    AngBtFlowAndRadii=angle_between(FlowVector, RadiiLineVector)
                    
                    HistTest.append(mag[GridC][GridR])
                    HistTestAng.append(AngBtFlowAndRadii)
                    
                    NewValue=(mag[GridC][GridR]) 
    # =============================================================================
    #               Change 360/bin size here
    # =============================================================================
                    NewIndex = math.floor(AngBtFlowAndRadii / 36)
                  #  NewIndex = AngBtFlowAndRadii // 15
                    NewHist[NewIndex] += NewValue
                        
                    if (mag[GridC][GridR]> MagThrushold):  
                        if (AngBtFlowAndRadii > 180):
                            #Clockwise
                            RelativeAng.append(1)
                        else:
                             #CounterClockwise
                            RelativeAng.append(0)
                    else:
                        SingleNoRotation+=1
                            
            URAng = np.sum(RelativeAng)        
            CriticalN = int(lsize) * int(lsize)/2 - SingleNoRotation
     #######################################################################
     #Display two histograms of the seletected cell 
     #######################################################################
            hist1,bins1 = np.histogram(HistTest, bins=20,range=(0,5))                    
            AllMag.append(hist1)
            hist2,bins2=np.histogram( HistTestAng,bins=24)
            AllAng.append(hist2)
            NewHistLong.append(NewHist)
        
     #######################################################################           
            if SingleNoRotation>=(int(lsize) * int(lsize)/2): #if no rotation
                URCellRotationD.append('NR')
                
            else:
                           
                if URAng < CriticalN:
                    URCellRotationD.append(0)
                else:
                    URCellRotationD.append(1)
       
        StepWiseResult.append(URCellRotationD)
        NoRotationDetermin.append(NoRotation)
                 #Replace the prvs with the current frame
        ret, frame = cap.read() #Get the new frame
        URCellRotationD = []   #Clear the recording list
        HistAngShown.append(HistTestAng)
        HistMagShown.append(HistTest)
        HoldNew.append(NewHistLong)
      #  if count==framenum:
       #     break 
        
    
    
    #Stepwise determination of the results  
        StepWiseResultArray = np.asarray(StepWiseResult)
        NoRotationArray=np.asarray(NoRotationDetermin)
        
        Determ =len(StepWiseResultArray)
        
        RotatResult = []
        CWlist=[]
        for e in range(CellNumb):
            NRFeature=0
            CWFeature=0
            for x in range(StepWiseResultArray[:,e].size):
                if StepWiseResultArray[:,e][x]=='NR':
                    NRFeature+=1
                elif StepWiseResultArray[:,e][x]=='1':
                 
                    CWFeature+=1
        
            CCWFeature=Determ-NRFeature-CWFeature
            if NRFeature>=(Determ/2):
                RotatResult.append('NR')
            else:
        
                if abs(CCWFeature-CWFeature)<=COMPLEXPARA:
                    RotatResult.append('Complex')
                else:
                    if CCWFeature>CWFeature:
                        RotatResult.append('CCW')
                    else:
                        RotatResult.append('CW')
        
         #   print ('cell # {}, NRFeature {}, CWFeature {} - CCWFeature {} = {}'.format(e,NRFeature,CWFeature,CCWFeature,CWFeature-CCWFeature))
# =============================================================================
#         #Plot the results   
#         plt.figure(figsize=(20,10))
#         im = plt.imread(imagef)
#         implot = plt.imshow(im)
# =============================================================================
        
        # put a red dot, size 40, at 2 locations:
# =============================================================================
#         plt.scatter(XCoord, YCoord, c='r', s=10)
# =============================================================================
        MyAnswer=[]
        for m, txt in enumerate(RotatResult):
            plt.annotate(txt, (XCoord[m],YCoord[m]))
            MyAnswer.append([str(XCoord[m]),str(YCoord[m]),txt])
        
# =============================================================================
#         plt.show()
# =============================================================================
        
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
        
        with open('D:\mic3\datacollection0820_bin10.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            for i in range(len(WRITE)):
               
               writer.writerow(WRITE[i])
            
        
        csvFile.close()
        
    #    with open('D:\mic3\datacollection0808_bin30_xy0{}.csv'.format(videonum), 'a') as csvFile:
    #        writer = csv.writer(csvFile)
    #        for i in range(len(WRITE)):
    #           
    #           writer.writerow(WRITE[i])
    #        
    #    
    #    csvFile.close()
        
        
        prvs = nxt 
        AllAng=[]
        AllMag=[]
        HoldNew=[]
        count=count+1
    cap.release()   