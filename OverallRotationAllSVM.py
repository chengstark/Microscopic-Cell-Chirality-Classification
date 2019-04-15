# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:08:53 2018

@author: Yang Zhang

Use SVM to train the overall rotation direction

2.use three groups' single rotation direction to predict the single rotation direction of the test group
   and use the overall rotation direction of those three groups to predict the overall rotation direction 
   of the test group.

 Mimic the actual cross validation process

"""
from matplotlib import pyplot as plt
import scipy
import numpy as np
from sklearn.svm import LinearSVC
import csv
from sklearn.svm import NuSVC
from sklearn.svm import SVC
# =============================================================================
# Import all Fiji-aided labelling results
# =============================================================================
OVERALL_LABEL_XY01=['CW','CCW','CW','CW','CW','CW','CCW','CW','CCW','CCW','CCW','CCW','CCW','CW','CW','CW','CCW','CW']
OVERALL_LABEL_XY02=['CW','CCW','CCW','CW','CCW','CCW','CW','CW','CCW','CW','NR','CCW','CW','CW','CCW']
OVERALL_LABEL_XY03=['CCW','CCW','CW','CCW','CW','CW','CW','NR','CCW','CCW']
OVERALL_LABEL_XY04=['CCW','CCW','CCW','CCW','CW','CCW','CW','CCW','CCW']
LABLE = []
for i in [1,2,3,4]:
    LabelFile='D:\mic3\FijiAidXY0{}.csv'.format(i)
    with open(LabelFile, newline='') as csvfile:
        
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            
            if row:
                a=[x.strip() for x in row[0].split(',')]
                if a[0]!='':
                    LABLE.append([x.strip() for x in row[0].split(',')])
OVERALLLABLE = OVERALL_LABEL_XY01 + OVERALL_LABEL_XY02 + OVERALL_LABEL_XY03 + OVERALL_LABEL_XY04
for i in range(len(OVERALLLABLE)):
    LABLE[i].append(OVERALLLABLE[i])

DATA=[]

with open('D:\mic3\seq0809_bin30.csv', newline='') as csvfile:
    
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        
        if row:
            a=[x.strip() for x in row[0].split(',')]
            if a[0]!='':
                DATA.append([x.strip() for x in row[0].split(',')])
# =============================================================================
# Training
# =============================================================================
TOTALNUMOVERALL=52
TRAININGNUMOVERALL=int(TOTALNUMOVERALL*3/4)
TOTALNUM=52*24
TRAININGNUM=int(TOTALNUM*3/4)
BINSIZE = 30
LAST=[]
# =============================================================================
# NOTE:
# if change the 1000 -1000 and 0 to be 0,1,2, the results change a lot
# =============================================================================
def Convert(Y):
    A=[]
    for i in range(Y.shape[0]):
        Answer=[]
        for j in range(Y.shape[1]):
            if Y[i][j]=='CCW':
                Answer.append(1000)
            elif Y[i][j]=='CW':
                Answer.append(-1000)
            else:
                Answer.append(0)
        A.append(Answer)
    return np.asarray(A)

for w in range(4):
        
    data_OVERALL=np.asarray(LABLE[:TRAININGNUMOVERALL])
    TrainingX_OVERALL=data_OVERALL
    TrainingX_OVERALL=np.delete(TrainingX_OVERALL,24,1)
    TrainingX_OVERALL=Convert(TrainingX_OVERALL)
       
    
    TrainingYY_OVERALL=data_OVERALL[...,24]
    
# =============================================================================
# Train single frame  
# =============================================================================
    TrainX_Single = []
    data=np.asarray(DATA[:TRAININGNUM])
    TrainingX=data
    TrainingX=np.delete(TrainingX,BINSIZE,1)
    TrainingX=TrainingX.astype(float)
    for i in range(TrainingX.shape[0]):
        TrainingX[i] = [number/scipy.linalg.norm(TrainingX[i]) for number in TrainingX[i]]
        
    
    TrainingY=data[...,BINSIZE]
    
    Test=np.asarray(DATA[TRAININGNUM:])
    TestX=Test
    TestX=np.delete(TestX,BINSIZE,1).astype(float)
    for i in range(TestX.shape[0]):
        TestX[i] = [number/scipy.linalg.norm(TestX[i]) for number in TestX[i]]
    
        
    TestY=Test[...,BINSIZE]
    #SELECT MODEL and train,random_state=0
    clf_single = SVC(kernel='sigmoid')
    clf_single.fit(TrainingX, TrainingY)
    
    count1=0
    temp=[]
    for i in range(TestX.shape[0]):
        count1+=1
        answer=clf_single.predict([TestX[i]])
        temp.append(answer[0])
        
        if count1==24:
            TrainX_Single.append(np.asarray(temp))
            temp=[]
            count1=0
            
    TrainX_Single=Convert(np.asarray(TrainX_Single))

# =============================================================================
#    end training single frame
#    Test_overall is the ground truth
# =============================================================================
    Test_OVERALL=np.asarray(LABLE[TRAININGNUMOVERALL:])

        
    TestY_OVERALL=Test_OVERALL[...,24]
    #SELECT MODEL and train,random_state=0
    clf = SVC(kernel='rbf')
    clf.fit(TrainingX_OVERALL, TrainingYY_OVERALL) 
    
    TestResult=[]
    a=0
    correct=0
    for i in range(TrainX_Single.shape[0]):
        a+=1
        #TEST by usiing the prediction given by the previous traing
        answer=clf.predict([TrainX_Single[i]])
       
        if answer[0] == TestY_OVERALL[i]:
            correct+=1
        else:
            print ('Prediction: {}, Answer: {}'.format(answer[0],TestY_OVERALL[i]))
        TestResult.append(answer)
        
      
    LAST.append('{}'.format(float(correct)/float(a)))
    LABLE=LABLE[39:]+LABLE[:39]
    DATA=DATA[39*24:]+DATA[:39*24]
for i in range(4):
    print(LAST[i])