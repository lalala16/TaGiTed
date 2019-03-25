__author__ = 'smileyk01'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2

def plot_learning_curve_yk_sup_single(rangC,rangD,PTrainSet,PTestSet):

    plt.figure()

    DMaxCPR=[]

    for j in range(rangD.shape[0]):
        phens=rangD[j]

        tX=pd.read_csv(path_experiment+'trainingX_'+str(phens)+'phens_ttapr_100outIter.csv',delim_whitespace=True,header=None)
        print tX.shape
        tY=pd.read_csv(path_experiment+'trainingY_'+str(phens)+'phens_ttapr_100outIter.csv',header=None)
        tY=np.array(tY).ravel()
        print len(tY)

        eX=pd.read_csv(path_experiment+'testX_'+str(phens)+'phens_ttapr_100outIter.csv',delim_whitespace=True,header=None)
        print eX.shape
        eY=pd.read_csv(path_experiment+'testY_'+str(phens)+'phens_ttapr_100outIter.csv',header=None)
        eY=np.array(eY).ravel()
        print len(eY)
        #tX=tX.fillna(1)

        #eX=eX.dropna(axis=0,how='any')
        #eY=eY[list(eX.index)]
        eX=eX.fillna(1)
        tXNorm=preprocessing.normalize(tX,norm='l2')
        eXNorm=preprocessing.normalize(eX,norm='l2')

        trainX=tXNorm[GoodIndex80]
        #trainX=tXNorm
        #print trainX.shape
        trainY=tY[GoodIndex80]
        #trainY=tY
        #print trainY.shape
        testX=eXNorm
        #print testX.shape
        testY=eY
        #print testY.shape

        plt.subplot(2,rangD.shape[0]/2+1,j+1)
        Ttitle = "Logistic Regression"+'PSize:'+str(rangD[j])
        title=Ttitle
        plt.title(title)

        plt.xlabel("C")
        plt.ylabel("Score")
        DMeanCPR=np.array([])

        for i in range(rangC.shape[0]):
            estimator = LogisticRegression(penalty='l2',C=rangC[i])
            #estimator = SVC(kernel="linear", C=rangC[i])
            estimator.fit(trainX,np.array(trainY).ravel())
            result =estimator.predict_proba(testX)[:,1]
            test_scores_mean=roc_auc_score(testY, result)
            DMeanCPR=np.append(DMeanCPR,test_scores_mean)

        DMaxCPR.append(np.max(DMeanCPR))

        plt.grid()
        LogRangC=[math.log(i) for i in rangC]
        plt.plot(LogRangC, DMeanCPR, 'o-', color="r",
                 label="CPR_AUC score")
        plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    Ttitle = "Logistic Regression"
    plt.title(Ttitle)
    plt.xlabel("Phenos")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(rangD, DMaxCPR, 'o-', color="r",
                 label="CPRP_AUC score")
    plt.legend(loc="lower right")
    plt.show()

def plot_learning_curve_yk_sup(rangC,rangD,PTrainSet,PTestSet):

    plt.figure()
    PTrainSet=list(set(PTrainSet)&GoodIndex8)
    RDX=pd.read_csv(path_experiment+'rawdataX_100phens_Uapr_100outIter.csv',delim_whitespace=True,header=None)
    print RDX.shape
    RDY=pd.read_csv(path_experiment+'rawdataY_100phens_Uapr_100outIter.csv',header=None)
    RDY=np.array(RDY).ravel()
    print len(RDY)
    RDXNorm=preprocessing.normalize(RDX,norm='l2')
    trainRDX=RDXNorm[np.array(PTrainSet)]
    #print trainRDX.shape
    trainRDY=pd.DataFrame(RDY).ix[np.array(PTrainSet)]
    #print trainRDY.shape
    testRDX=RDXNorm[np.array(PTestSet)]
    #print testRDX.shape
    testRDY=pd.DataFrame(RDY).ix[np.array(PTestSet)]

    DMaxCPR=[]
    DMaxRAW=[]
    DMaxRAWPCA=[]
    for j in range(rangD.shape[0]):
        phens=rangD[j]

        SKB=SelectKBest(chi2, k=phens)
        tRDXS = SKB.fit_transform(trainRDX, np.array(trainRDY).ravel())
        #print testRDY.shape
        teRDXS=SKB.transform(testRDX)

        pca = PCA(n_components=phens)
        RDXNormPCA = pca.fit(trainRDX).transform(trainRDX)
        teRDXNormPCA=pca.transform(testRDX)

        tX=pd.read_csv(path_experiment+'trainingX_'+str(phens)+'phens_ttapr_100outIter.csv',delim_whitespace=True,header=None)
        print tX.shape
        tY=pd.read_csv(path_experiment+'trainingY_'+str(phens)+'phens_ttapr_100outIter.csv',header=None)
        tY=np.array(tY).ravel()
        print len(tY)

        eX=pd.read_csv(path_experiment+'testX_'+str(phens)+'phens_ttapr_100outIter.csv',delim_whitespace=True,header=None)
        print eX.shape
        eY=pd.read_csv(path_experiment+'testY_'+str(phens)+'phens_ttapr_100outIter.csv',header=None)
        eY=np.array(eY).ravel()
        print len(eY)
        #tX=tX.fillna(1)

        eX=eX.dropna(axis=0,how='any')
        eY=eY[list(eX.index)]
        #eX=eX.fillna(1)
        tXNorm=preprocessing.normalize(tX,norm='l2')
        eXNorm=preprocessing.normalize(eX,norm='l2')

        trainX=tXNorm[GoodIndex80]
        #trainX=tXNorm
        #print trainX.shape
        trainY=tY[GoodIndex80]
        #trainY=tY
        #print trainY.shape
        testX=eXNorm
        #print testX.shape
        testY=eY
        #print testY.shape

        plt.subplot(2,rangD.shape[0]/2+1,j+1)
        Ttitle = "Logistic Regression"+'PSize:'+str(rangD[j])
        title=Ttitle
        plt.title(title)

        plt.xlabel("C")
        plt.ylabel("Score")
        DMeanCPR=np.array([])
        DMeanRAW=np.array([])
        DMeanRAWPCA=np.array([])

        for i in range(rangC.shape[0]):
            estimator = LogisticRegression(penalty='l2',C=rangC[i])
            #estimator = SVC(kernel="linear", C=rangC[i])
            estimator.fit(trainX,np.array(trainY).ravel())
            result =estimator.predict_proba(testX)[:,1]
            test_scores_mean=roc_auc_score(testY, result)
            DMeanCPR=np.append(DMeanCPR,test_scores_mean)

            estimator = LogisticRegression(penalty='l2',C=rangC[i])
            #estimator = GaussianNB()
            #estimator = SVC(gamma=0.001)
            estimator.fit(tRDXS,np.array(trainRDY).ravel())
            result =estimator.predict_proba(teRDXS)[:,1]
            test_scores_mean=roc_auc_score(testRDY, result)
            DMeanRAW=np.append(DMeanRAW,test_scores_mean)

            estimator = LogisticRegression(penalty='l2',C=rangC[i])
            #estimator = GaussianNB()
            #estimator = SVC(gamma=0.001)
            estimator.fit(RDXNormPCA,np.array(trainRDY).ravel())
            result =estimator.predict_proba(teRDXNormPCA)[:,1]
            test_scores_mean=roc_auc_score(testRDY, result)
            DMeanRAWPCA=np.append(DMeanRAWPCA,test_scores_mean)


        DMaxCPR.append(np.max(DMeanCPR))
        DMaxRAW.append(np.max(DMeanRAW))
        DMaxRAWPCA.append(np.max(DMeanRAWPCA))

        plt.grid()
        LogRangC=[math.log(i) for i in rangC]
        plt.plot(LogRangC, DMeanCPR, 'o-', color="r",label="CPR_AUC score")
        plt.plot(LogRangC, DMeanRAW, 'o-', color="g",label="RAW+FSK_AUC score")
        plt.plot(LogRangC, DMeanRAWPCA, 'o-', color="b",label="RAW+PCA_AUC score")
        plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    Ttitle = "Logistic Regression"
    plt.title(Ttitle)
    plt.xlabel("Phenos")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(rangD, DMaxCPR, 'o-', color="r",
                 label="CPRP_AUC score")
    plt.plot(rangD, DMaxRAW, 'o-', color="g",
             label="RAW+FSK_AUC score")
    plt.plot(rangD, DMaxRAWPCA, 'o-', color="b",
             label="RAW+PCA_AUC score")
    plt.legend(loc="lower right")
    plt.show()

def generatePLT(oo,PTrainSet,PTestSet):
    #estimator = GaussianNB()
    #estimator = SVC(gamma=0.001)
    rangC=np.array([0.001,0.01,0.1,1,10,100])
    rangD=np.array([30,50,80,100,120,150,180])
    #plot_learning_curve_yk_sup(rangC,rangD,PTrainSet,PTestSet)
    plot_learning_curve_yk_sup_single(rangC,rangD,PTrainSet,PTestSet)
    print oo

path_experiment='D:/yk/py_workspace/xiamendata/experiment1/'
path_notebook='D:/yk/notebook/'
path_extract='D:/yk/py_workspace/CleanExtract/'
