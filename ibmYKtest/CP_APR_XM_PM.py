__author__ = 'yangkai01'
# coding: utf-8

import tensor;
import sptensor;
import numpy as np;
import pandas as pd
import csv
import CP_APR
import ktensor
import predictionModel

#set data dirs
data_dir = 'E:/test_project_python/XiamenData/'
data_notebook='D:/yk/notebook/'
data_extract='D:/yk/py_workspace/CleanExtract/'

# from spase representation file to spase tensor
x=pd.read_csv(data_dir+'validationTensor_3412.csv').drop(['PID'],axis=1)
#print x
matrixIndex=np.array(x.ix[:,:3].as_matrix(),dtype='int')
matrixValue=np.array(x.ix[:,3].as_matrix(),dtype='int').reshape((matrixIndex.shape[0],1))
matrixSize=np.array([3412,247,816])

#print matrixValue
re=pd.read_csv(data_dir+'vadationResult_3412.csv')
#print re
Y=np.array(re.ix[:,'InHosLabel'])
#print Y


X=sptensor.sptensor(matrixIndex,matrixValue,matrixSize)
#print X.subs
#print X.subs.shape

'''
demoX=pd.read_csv(data_notebook+'demoF.csv')
demoX.index=demoX.ix[:,0]
demoX=np.array(demoX.ix[:,1:])
#demoX=demoX[:,1:]

print(demoX.shape)
#print demoX[:3]
'''

goNum=[10,30,50,80,100,125,150,180,200]
for i in range(len(goNum)):
    phennum=goNum[i]
    pm = predictionModel.predictionModel(X,X.subs, Y, phennum)
    pm.evaluatePredictionAUC_1(str(phennum)+'phens_Uapr_100outIter')

csvfile=file(data_dir+'probablity','wb')
writer=csv.writer(csvfile)
csvfile.close()


