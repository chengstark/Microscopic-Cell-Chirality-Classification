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
from sklearn.svm import SVC
LOCALPARA=2
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

for i in [1]:
    videonum= i
    ############################################################
    #Choose the video to be analyzed, and set the parameter    #
    ############################################################
    videof = r'D:\CellRotation\VideoDATA\XY0{}_video.avi'.format(videonum)
    imagef = r'D:\CellRotation\VideoDATA\XY0{}_video.avi_1.tif'.format(videonum)

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
    cv2.imshow("Keypoints", im_with_keypoints)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        savepath='D:\CellRotation\VideoDATARsults\20180904_XY0{}.png'.format(i)
        cv2.imwrite(savepath, im_with_keypoints)
        cv2.destroyAllWindows()


    
    XCoord, YCoord, Size = CoordinatesD(keypoints)
    CellNumb = len(XCoord)
# =============================================================================
#     fig,ax = plt.subplots(1,figsize=(30,25))
# 
# # Display the image
#     ax.imshow(prvs)
#     for p in range(CellNumb):
#     # Create a Rectangle patch, Add the patch to the Axes
#         left=(XCoord[p]-int(Size[p])/2)-2
#         bottom=(YCoord[p]-int(Size[p])/2)-2
#         lsize=Size[p]+4
#         rect = patches.Rectangle((left,bottom),lsize,lsize,linewidth=1,edgecolor='b',facecolor='none')
#         ax.add_patch(rect)
#         ax.plot(left,bottom,'ro')
#         ax.plot(XCoord[p],YCoord[p],'go')
#         ax.annotate(str(p), (left,bottom))
#     plt.show()
# =============================================================================
    AllList=[]
    for i in range(CellNumb):
        AllList.append([])

   # ret, frame = cap.read()
    count=1
    
    HoldNew=[]
            
    FrameIND=np.linspace(5,120,24)

    while(ret):
        if count in FrameIND:
            if count ==5:
                ret, frame = cap.read()
                

            print ('Frame# {} is processed'.format(count))
            #
# =============================================================================
#             cap = cv2.VideoCapture(0)
#             ret, frame1 = cap.read()
#             
#             prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# =============================================================================

                   
                   
            nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,nxt, flow=None, pyr_scale=0.5,levels=3,winsize=5,iterations=20, poly_n=10,poly_sigma=1.1,flags=0)        
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1],angleInDegrees=1)
            
            
            
            hsv = np.zeros_like(frame)
            hsv[...,1] = 255
            #Displaying the color coding diagram to visualize the optical flow
           # flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            cv2.imshow('Optical Flow Aura',bgr)
            k = cv2.waitKey(0) & 0xFF
            if  k == 27:  # press q to quit
      
                savename=r'D:\mic3\aura\{}_{}.png'.format(videonum,count)
                cv2.imwrite(savename,bgr)
                cv2.destroyAllWindows()
                    
                # When everything done, release the capture
            fig,ax = plt.subplots(1,figsize=(30,25))   
            ax.imshow(bgr)
            for p in range(CellNumb):
         
                left=(XCoord[p]-int(Size[p])/2)-2
                bottom=(YCoord[p]-int(Size[p])/2)-2
                lsize=Size[p]+4
                rect = patches.Rectangle((left,bottom),lsize,lsize,linewidth=1,edgecolor='b',facecolor='none')
                ax.add_patch(rect)
                ax.plot(left,bottom,'ro')
                ax.plot(XCoord[p],YCoord[p],'go')
                ax.annotate(str(p), (left,bottom))
            plt.show()
                   
# =============================================================================
#             if count == 5:
#                 step=4
#                 h, w = nxt.shape[:2]
#                 yy, xx = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
#                 y = yy.astype(int)
#                 x=xx.astype(int)
#                 fx, fy = flow[y,x].T
#                 m = np.bitwise_and(np.isfinite(fx), np.isfinite(fy))
#                 lines = np.vstack([x[m], y[m], x[m]+fx[m], y[m]+fy[m]]).T.reshape(-1, 2, 2)
#                 lines = np.int32(lines + 0.5)
#                 vis = cv2.cvtColor(nxt, cv2.COLOR_GRAY2BGR)
#                 cv2.polylines(vis, lines, 0, (0, 255, 0))
#                 for (x1, y1), (x2, y2) in lines:
#                     cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#                 
#                 cv2.imshow('image',vis)
#                 k = cv2.waitKey(0) & 0xFF
#                 if k == 27:         # wait for ESC key to exit
#                     cv2.destroyAllWindows()
# =============================================================================
            #Optical Flow detection            
            NewHistLong=[]
        
            for i in range(CellNumb):
        # =============================================================================
        #         Change bin size here
        # =============================================================================
                NewHist=[0]*24
    
                #Get the coordinate of the center point and the four corner points of a window
                #in the whole image array.
                CenterX=XCoord[i]
                CenterY=YCoord[i]
                
                left=int(XCoord[i]-Size[i]/2-LOCALPARA)
                bottom=int(YCoord[i]-Size[i]/2-LOCALPARA)
                lsize=Size[i]+LOCALPARA*2
    
                for r in range(lsize):
                    for c in range(lsize):
                        #The vector from the current point to 
                        GridR=left+r
                        GridC=bottom+c
                        RadiiLineVector = (CenterX-GridR,CenterY-GridC)
                        FlowVector = (flow[GridC][GridR][0],flow[GridC][GridR][1])
                        AngBtFlowAndRadii=angle_between(FlowVector, RadiiLineVector)
                        
                        NewValue=(mag[GridC][GridR]) 
        # =============================================================================
        #               Change 360/bin size here
        # =============================================================================
                        NewIndex = math.floor(AngBtFlowAndRadii / 15)
                      #  NewIndex = AngBtFlowAndRadii // 15
                        NewHist[NewIndex] += NewValue
                            
                
                AllList[i].append(NewHist)
            
         #######################################################################           
                
            HoldNew.append(NewHistLong)        
            prvs = nxt 

        
    #    with open('D:\mic3\datacollection0808_bin30_xy0{}.csv'.format(videonum), 'a') as csvFile:
    #        writer = csv.writer(csvFile)
    #        for i in range(len(WRITE)):
    #           
    #           writer.writerow(WRITE[i])
    #        
    #    
    #    csvFile.close()
        print (count)
        count=count+1
        ret, frame = cap.read()  

        
       
        WRITE=[]
        for i in range(len(AllList)):
            for j in range(len(AllList[i])):
                WRITE.append(AllList[i][j])
        
        with open('D:\mic3\Trial1_0929.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            for i in range(len(WRITE)):
               
               writer.writerow(WRITE[i])
            
        
        csvFile.close()
    cap.release()