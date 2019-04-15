# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 20:36:40 2018

@author: Yang Zhang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 23:30:12 2018

@author: Yang Zhang


Trying to train data getting from different bin size
"""
from matplotlib import pyplot as plt
import scipy
import numpy as np
from sklearn.svm import LinearSVC
import csv
from sklearn.svm import NuSVC
from sklearn.svm import SVC
'''
'Other than "complex", there are 52 cells with lables (CW,CCW,NR)
'13 cells for a group
'''

##########################################
#   GLOBAL VARIABLES
#################################
DATA=[]
OVERALL_LABEL_XY01=['CW','CCW','CW','CW','CW','CW','CCW','CW','CCW','CCW','CCW','CCW','CCW','CW','CW','CW','CCW','CW']
OVERALL_LABEL_XY02=['CW','CCW','CCW','CW','CCW','CCW','CW','CW','CCW','CW','NR','CCW','CW','CW','CCW']
OVERALL_LABEL_XY03=['CCW','CCW','CW','CCW','CW','CW','CW','NR','CCW','CCW']
OVERALL_LABEL_XY04=['CCW','CCW','CCW','CCW','CW','CCW','CW','CCW','CCW']
xy01=len(OVERALL_LABEL_XY01)
xy02=len(OVERALL_LABEL_XY02)
xy03=len(OVERALL_LABEL_XY03)
xy04=len(OVERALL_LABEL_XY04)
########################################
# Read data from .csv file    'D:\mic3\datacollection_3rdFrame.csv'
##########################################


with open('D:\mic3\datacollection0820_bin10.csv', newline='') as csvfile:
    
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        
        if row:
            a=[x.strip() for x in row[0].split(',')]
            if a[0]!='':
                DATA.append([x.strip() for x in row[0].split(',')])
#%%
XYSeq=[]
#+432  + 795
XY=[xy01,xy02,xy03,xy04]
for item in XY:
    counter=0
    with open('D:\mic3\seq0820_bin10.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        for j in range(item):
            for i in range(0,24):
                print(i*item+j)
                writer.writerow(DATA[i*item+j])
                DATA[i*item+j]=[0]
                counter+=1
           # print('xxxx')
    for u in range(counter):
        DATA.remove([0])
    csvFile.close()
    
    #    with open('D:\mic3\datacollection0808_bin30_xy0{}.csv'.format(videonum), 'a') as csvFile:
    #        writer = csv.writer(csvFile)
    #        for i in range(len(WRITE)):
    #           
    #           writer.writerow(WRITE[i])
    #        
    #    
    #    csvFile.close()
                
#%%

#############################################
#assigned data into groups 13cells/group
#   4 groups total
#   3 groups (39) for training, and 1 group (13) as testing group
#############################################
# =============================================================================
# 
# def Convert(Y):
#     Answer=[]
#     for i in range(Y.shape[0]):
#         if Y[i]=='CCW':
#             Answer.append(1)
#         elif Y[i]=='CW':
#             Answer.append(2)
#         else:
#             Answer.append(3)
#     return Answer
# ################################################
# #    Start assigning data into training set and testing set
# #####################################################  
# TOTALNUM=52*24
# BINSIZE=24
# TRAININGNUM=int(TOTALNUM*3/4)
# LAST=[]
# AllTest=[]
# for w in range(4):
#     
#     data=np.asarray(DATA[:TRAININGNUM])
#     TrainingX=data
#     TrainingX=np.delete(TrainingX,BINSIZE,1)
#     TrainingX=TrainingX.astype(float)
#     for i in range(TrainingX.shape[0]):
#         TrainingX[i] = [number/scipy.linalg.norm(TrainingX[i]) for number in TrainingX[i]]
#         
#     
#     TrainingY=data[...,BINSIZE]   
#     
#     
#     Test=np.asarray(DATA[TRAININGNUM:])
#     TestX=Test
#     TestX=np.delete(TestX,BINSIZE,1).astype(float)
#     for i in range(TestX.shape[0]):
#         TestX[i] = [number/scipy.linalg.norm(TestX[i]) for number in TestX[i]]
#     
#         
#     TestY=Test[...,BINSIZE]
#     #SELECT MODEL and train,random_state=0
#     clf = SVC(kernel='sigmoid')
#     clf.fit(TrainingX, TrainingY) 
#     
#     #     
#     TestResult=[]
#     a=0
#     correct=0
#     for i in range(TestX.shape[0]):
#         a+=1
#         answer=clf.predict([TestX[i]])
#         if answer[0] == TestY[i]:
#             correct+=1
#         TestResult.append(answer[0])
#         
#         # plt.figure(figsize=(20,10))
#         # plt.plot( np.arange(24),TestX[i])
#         # plt.xlabel(TestY[i])
#         # plt.show()
#       #  print ('SVM determine the cell as {}, ANSWER: {} '.format(answer,TestY[i]))
#     
#   #  print ('correctness is {}'.format(float(correct)/float(a)))   
#     LAST.append('{}'.format(float(correct)/float(a)))
#     DATA=DATA[39:]+DATA[:39]
#     AllTest.append(TestResult)
#     
# for w in range(4):
#     print (LAST[w])
# SeqResult=[]
# for i in [3,2,1,0]:
#     SeqResult+=AllTest[i]
# =============================================================================
    