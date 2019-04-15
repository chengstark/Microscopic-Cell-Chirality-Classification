# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 23:30:12 2018

@author: Yang Zhang
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


########################################
# Read data from .csv file    'D:\mic3\datacollection_3rdFrame.csv'
##########################################


with open('D:\mic3\datacollection0802.csv', newline='') as csvfile:
    
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        
        if row:
            a=[x.strip() for x in row[0].split(',')]
            if a[0]!='':
                DATA.append([x.strip() for x in row[0].split(',')])
                '''
for i in range(10):
    plt.figure(figsize=(20,10))
    #x=list(range(24))
    y=np.asarray(DATA[i][:24]).astype(float)
    
    plt.plot(np.arange(24), y)
    plt.show()
    '''
#%%
#############################################
#assigned data into groups 13cells/group
#   4 groups total
#   3 groups (39) for training, and 1 group (13) as testing group
#############################################

def Convert(Y):
    Answer=[]
    for i in range(Y.shape[0]):
        if Y[i]=='CCW':
            Answer.append(1)
        elif Y[i]=='CW':
            Answer.append(2)
        else:
            Answer.append(3)
    return Answer
################################################
#    Start assigning data into training set and testing set
#####################################################  
TOTALNUM=52*24

TRAININGNUM=int(TOTALNUM*3/4)
LAST=[]
for i in range(4):
    
    data=np.asarray(DATA[:TRAININGNUM])
    TrainingX=data
    TrainingX=np.delete(TrainingX,24,1)
    TrainingX=TrainingX.astype(float)
    for i in range(TrainingX.shape[0]):
        TrainingX[i] = [number/scipy.linalg.norm(TrainingX[i]) for number in TrainingX[i]]
        
    
    TrainingYY=data[...,24]
    TrainingY=Convert(TrainingYY)
    
    
    
    Test=np.asarray(DATA[TRAININGNUM:])
    TestX=Test
    TestX=np.delete(TestX,24,1).astype(float)
    for i in range(TestX.shape[0]):
        TestX[i] = [number/scipy.linalg.norm(TestX[i]) for number in TestX[i]]
    
        
    TestY=Convert(Test[...,24])
    #SELECT MODEL and train,random_state=0
    clf = SVC(kernel='rbf')
    clf.fit(TrainingX, TrainingY) 
    
    #     
    TestResult=[]
    a=0
    correct=0
    for i in range(TestX.shape[0]):
        a+=1
        answer=clf.predict([TestX[i]])
        if answer[0] == TestY[i]:
            correct+=1
        TestResult.append(answer)
        
        # plt.figure(figsize=(20,10))
        # plt.plot( np.arange(24),TestX[i])
        # plt.xlabel(TestY[i])
        # plt.show()
      #  print ('SVM determine the cell as {}, ANSWER: {} '.format(answer,TestY[i]))
    
  #  print ('correctness is {}'.format(float(correct)/float(a)))   
    LAST.append('correctness is {}'.format(float(correct)/float(a)))
    DATA=DATA[39:]+DATA[:39]
    
for w in range(4):
    print (LAST[w])