# -*- coding: utf-8 -*-
"""
@author: Zhicheng Fang, Pingkun Yan
"""

import numpy as np
import cv2

# %%

videof = '10x_XY04_video.avi'
imagef = '10x_XY04_video_8bit.tif'
Resultf = 'Result04.xlsx'

cap = cv2.VideoCapture(videof)
ret, frame = cap.read()
prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
StepWiseResult = []

#
# Set up the detector with default parameters.
#
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
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.3
    
#
# Create a detector with the parameters
#
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(frame)

# Detect Center and Size

def CoordinatesD(keypoints):
    '''
    '''
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
# %%%
while(1):
    ret, frame = cap.read()
    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Optical Flow detection
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 5, 20, 10, 1.1, 0)

    #Angle calculation for Optical flow vector
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=1)

    #Define the optial flow direction detection #Method1
    URCellRotationD = []
    for i in range(CellNumb):
        URAng = ang[int(YCoord[i]) + int(Size[i])/2, int(XCoord[i])]
        if 90 < URAng < 270:
            URCellRotationD.append(0)
        else:
            URCellRotationD.append(1)
    StepWiseResult.append(URCellRotationD)

    k = cv2.waitKey(24) & 0xff
    if k == 20:
        break
    elif k == ord('s'):
        prvs = next
        
cap.release()
  #%% 
#Stepwise determination of the results  
StepWiseResultArray = np.asarray(StepWiseResult)
Determ =len(StepWiseResultArray)/2
DetermUp = Determ + 5.5
DetermLow = Determ - 5.5
RotatResult = []
for e in range(CellNumb):
    CWFeature = np.sum(StepWiseResultArray[:,e])
    print(CWFeature)
    if CWFeature > DetermUp:
        RotatResult.append('CW')
    if CWFeature < DetermLow:
        RotatResult.append('CCW')
    if DetermLow <= CWFeature <= DetermUp:
        RotatResult.append('Complex')
        
#Plot the results   
import matplotlib.pyplot as plt
im = plt.imread(imagef)
implot = plt.imshow(im)

# put a red dot, size 40, at 2 locations:
plt.scatter(XCoord, YCoord, c='r', s=10)

for m, txt in enumerate(RotatResult):
    plt.annotate(txt, (XCoord[m],YCoord[m]))

plt.show()

#Save the data to excel file
import pandas as pd

# Create a Pandas dataframe from some data.
dfx = pd.DataFrame({'X-Coordinate of Center': XCoord})
dfy = pd.DataFrame({'Y-Coordinate of Center': YCoord})
dfr = pd.DataFrame({'Analysis': RotatResult})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(Resultf, engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
dfx.to_excel(writer, sheet_name='Sheet1')
dfy.to_excel(writer, sheet_name='Sheet1', startcol=3)
dfr.to_excel(writer, sheet_name='Sheet1', startcol=6)

# Close the Pandas Excel writer and output the Excel file.
writer.save()     