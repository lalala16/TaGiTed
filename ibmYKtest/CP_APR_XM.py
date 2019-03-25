__author__ = 'yangkai01'
# coding: utf-8

import sptensor
import numpy as np
import pandas as pd
import csv
import CP_APR
import ktensor
import predictionModel

#set data dirs
data_dir = 'E:/test_project_python/XiamenData/'

trainX=pd.read_csv(data_dir+'SubTrainTensor_2387.csv',index_col=0,)
trainX=trainX.drop(['PID'],axis=1)
trainXIndex=np.array(trainX.ix[:,:3].as_matrix(),dtype='int')
trainXValue=np.array(trainX.ix[:,3].as_matrix(),dtype='int').reshape((trainXIndex.shape[0],1))
trainXSize=np.array([2387,247,816])
trainTensor=sptensor.sptensor(trainXIndex,trainXValue,trainXSize)

testX=pd.read_csv(data_dir+'SubTestTensor_1024.csv',index_col=0,)
testX=testX.drop(['PID'],axis=1)
testXIndex=np.array(testX.ix[:,:3].as_matrix(),dtype='int')
testXValue=np.array(testX.ix[:,3].as_matrix(),dtype='int').reshape((testXIndex.shape[0],1))
testXSize=np.array([1024,247,816])
testTensor=sptensor.sptensor(testXIndex,testXValue,testXSize)


trainRe=pd.read_csv(data_dir+'TrainResult_2387.csv')
trainY=trainRe.ix[:,-1].as_matrix()
trainY[trainY==0]=-1
print trainY.sum()


testRe=pd.read_csv(data_dir+'TestResult_1024.csv')
testY=testRe.ix[:,-1].as_matrix()
testY[testY==0]=-1
print testY.sum()


#goNum=[30,50,80,100,120,150,180,200]
goNum=[10,30,50,80,100,125,150,180,200]
for i in range(len(goNum)):
    phennum=goNum[i]
    pm = predictionModel.predictionModel(trainTensor,trainTensor.subs, trainY, phennum)
    if i==0:
        first=True
    else:
        first=False
    pm.evaluatePredictionAUC_4(str(phennum)+'phens_ttapr_100outIter',trainTensor, trainY,testTensor,testY)

csvfile=file(data_dir+'probablity','wb')
writer=csv.writer(csvfile)
csvfile.close()


