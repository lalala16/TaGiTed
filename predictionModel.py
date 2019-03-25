import itertools

import numpy as np
from scipy.optimize import nnls
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import NMF
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LogisticRegression
import CP_APR
import KLProjection
import cp_apr_demog
import cp_apr_logis, cp_apr_linear
import predictionTools
import sptenmat
import sptensor

def createRawFeatures(X):
    mode2Offset = X.shape[1]
    #print 'modeoffset'
    #print mode2Offset
    rawFeat = np.zeros((X.shape[0], mode2Offset+X.shape[2]))
    for k in range(X.subs.shape[0]):
        sub = X.subs[k,:]
        #if  sub[1]>=500:
            #print sub[0], sub[1]
        rawFeat[sub[0], sub[1]] = rawFeat[sub[0], sub[1]] + X.vals[k,0]
        #print sub[0],sub[1],sub[2]
        rawFeat[sub[0], mode2Offset + sub[2]] = rawFeat[sub[0], mode2Offset + sub[2]] + X.vals[k,0]
    return rawFeat

class predictionModel:
    X = None
    axisInfo = None
    Y = None
    R = 0
    samples = 0
    ttss = None
    innerIter = 10
    outerIter = 70
    rawFeatures = None
    pcaModel = None
    predModel = None
    nmfModel = None
    flatX = None

    data_dir = 'E:/test_project_python/XiamenData/'

    def __init__(self, X, XAxis, Y, R, outerIter=20, testSize=0.5, samples=10, seed=10):
        self.X = X
        self.axisInfo = np.array(XAxis[0], dtype="int")
        self.Y = Y
        self.R = R
        self.outerIter = outerIter
        self.samples = samples
        self.ttss = StratifiedShuffleSplit(Y,n_iter=samples, test_size=testSize, random_state=seed)
        self.rawFeatures = createRawFeatures(X)
        self.flatX =  sptenmat.sptenmat(X, [0]).tocsrmat() # matricize along the first mode
        self.pcaModel = RandomizedPCA(n_components=R)
        self.predModel = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        self.nmfModel = NMF(n_components=R, max_iter = self.outerIter, nls_max_iter = self.innerIter)
    
    @staticmethod
    def rebase(ids, subs):
        """ 
        Re-index according to the ordered array that specifies the new indices
        
        Parameters
        ------------
        ids : ordered array that embeds the location
        subs : the locations that need to be 'reindexed' according to the ids
        """
        idMap = dict(itertools.izip(ids, range(len(ids))))
        for k in range(subs.shape[0]):
            id = subs[k, 0]
            subs[k, 0] = idMap[id]
        return subs
        
    def findFactors(self, trainX, zeroThr=1e-4):
        """ Find the factor basis for this tensor """
        M, cpstats, mstats = CP_APR.cp_apr(trainX, R=self.R, maxiters=self.outerIter, maxinner=self.innerIter)
        M.normalize_sort(1)
        # zero out the small factors
        for n in range(M.ndims()):
            zeroIdx = np.where(M.U[n] < zeroThr)
            M.U[n][zeroIdx] = 0
        return KLProjection.KLProjection(M.U, self.R)
    
    def nmfTransform(self):
        """ Replace the existing numpy implementation to work on sparse tensor """
        W = np.zeros((self.flatX.shape[0], self.nmfModel.n_components_))
        for j in xrange(0, self.flatX.shape[0]):
            W[j, :], _ = nnls(self.nmfModel.components_.T, np.ravel(self.flatX.getrow(j).todense()))
        return W
        
    def evaluatePrediction(self):
        run = 0
        #output = np.zeros((1,7))
        output = np.zeros((1,4))
        for train, test in self.ttss:
            print "Evaluating Run:{0}".format(run)
            # get the indices for the training tensor
            trainShape = list(self.X.shape)
            trainShape[0] = len(train)
            trainX = tensorSubset(self.X, train, trainShape)
            trainY = self.Y[train]
            ## find the tensor factors for PTF-HT
            klp = self.findFactors(self.X)
            ## Get the reduced features for the data points
            ptfFeat = klp.projectSlice(self.X, 0)
            ## Calculate the PCA baseline
        #    self.pcaModel.fit(self.flatX[train, :].toarray())         #i have chaged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  +toarray()
        #    pcaFeat = self.pcaModel.transform(self.flatX.toarray())   #i have chaged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  +toarray()
            ## Calculate the NMF baseline
        #    self.nmfModel.fit(self.flatX[train, :])
        #    nmfFeat = self.nmfTransform()
            ## Evaluate the raw fit using logistic regression
            self.predModel.fit(self.rawFeatures[train, :], trainY)
            rawPred = self.predModel.predict_proba(self.rawFeatures[test,:])
            ## Evaluate the PCA fit using logistic regression
        #    self.predModel.fit(pcaFeat[train, :], trainY)
        #    pcaPred = self.predModel.predict_proba(pcaFeat[test,:])
            ## Evaluate the baseline features using logistic regression 
        #    self.predModel.fit(nmfFeat[train, :], trainY)
        #    basePred = self.predModel.predict_proba(nmfFeat[test,:])
            ## Evaluate the reduced fit using logistic regression
            #print(train)
            #print test
            #print('yangkai')
            #print ptfFeat

            #print(ptfFeat[test,:])
            self.predModel.fit(ptfFeat[train, :], trainY)
            ptfPred = self.predModel.predict_proba(ptfFeat[test,:])
            ## stack the tuples for storage
            testY = self.Y[test]
        #    temp = np.column_stack((np.repeat(run, len(testY)), self.axisInfo[test], rawPred[:, 1], pcaPred[:,1], basePred[:, 1], ptfPred[:,1], testY))
            temp = np.column_stack((np.repeat(run, len(testY)),  rawPred[:, 1], ptfPred[:,1], testY))
            output = np.vstack((output, temp))
            run = run + 1
        output = np.delete(output, (0), axis=0)
        return output

    def evaluatePredictionAUC(self):
            run = 0
            sumBaseAUC=0.0
            sumCprAUC=0.0
            for train, test in self.ttss:
                print "Evaluating Run:{0}".format(run)
                # get the indices for the training tensor
                trainShape = list(self.X.shape)
                trainShape[0] = len(train)
                trainX = tensorSubset(self.X, train, trainShape)
                trainY = self.Y[train]
                ## find the tensor factors for PTF-HT
                klp = self.findFactors(trainX)

                ## Get the reduced features for the data points
                ptfFeat = klp.projectSlice(self.X, 0)
                ## Evaluate the raw fit using logistic regression

                baseAUC, basePred = predictionTools.getAUC(self.predModel, self.rawFeatures, self.Y, train, test)
                cprAUC, cprPred = predictionTools.getAUC(self.predModel, ptfFeat, self.Y, train, test)
                sumBaseAUC+=baseAUC
                sumCprAUC+=cprAUC

                run = run + 1
            print sumBaseAUC/run
            print('**************************************')
            print sumCprAUC/run
            return sumBaseAUC/run,sumCprAUC/run
    def evaluatePredictionAUC_1(self,experCount):
        run = 0
        sumBaseAUC=0.0
        sumCprAUC=0.0
        MCPR, cpstats, mstats = CP_APR.cp_apr(self.X, self.R, maxiters=1, maxinner=7)

        #MCPR.normalize_sort(1)
        MCPR.redistribute(0)
        ## scale by summing across the rows
        totWeight = np.sum(MCPR.U[0], axis=1)
        zeroIdx = np.where(totWeight < 1e-100)[0]
        if len(zeroIdx) > 0:
            # for the zero ones we're going to evenly distribute
            evenDist = np.repeat(1.0 / self.R, len(zeroIdx)*self.R)
            MCPR.U[0][zeroIdx, :] = evenDist.reshape((len(zeroIdx), self.R))
            totWeight = np.sum(MCPR.U[0], axis=1)
        twMat = np.repeat(totWeight, self.R).reshape(self.X.shape[0], self.R)
        MCPR.U[0] = MCPR.U[0] / twMat

        rawXfile=self.data_dir+'experiment_runprecess/rawdataX_'+str(experCount)+'.csv'
        rawYfile=self.data_dir+'experiment_runprecess/rawdataY_'+str(experCount)+'.csv'
        cprXfile=self.data_dir+'experiment_runprecess/cprdataX_'+str(experCount)+'.csv'
        cprYfile=self.data_dir+'experiment_runprecess/cprdataY_'+str(experCount)+'.csv'
        np.savetxt(rawXfile,self.rawFeatures)
        np.savetxt(rawYfile,self.Y)
        np.savetxt(cprXfile, MCPR.U[0])
        np.savetxt(cprYfile,self.Y)

    def evaluatePredictionAUC_2(self,experCount,Demog):
        run = 0
        sumBaseAUC=0.0
        sumCprAUC=0.0
        lambda1=1
        lambda4=1
        DemoU=np.random.rand(self.R,Demog.shape[1])
        MCPR, cpstats, mstats = cp_apr_demog.cp_apr(self.X, self.R,Demog,DemoU,lambda1,lambda4, maxiters=40, maxinner=self.innerIter)
        MCPR.normalize_sort(1)

        ## scale by summing across the rows
        totWeight = np.sum(MCPR.U[0], axis=1)
        zeroIdx = np.where(totWeight < 1e-100)[0]
        if len(zeroIdx) > 0:
            # for the zero ones we're going to evenly distribute
            evenDist = np.repeat(1.0 / self.R, len(zeroIdx)*self.R)
            MCPR.U[0][zeroIdx, :] = evenDist.reshape((len(zeroIdx), self.R))
            totWeight = np.sum(MCPR.U[0], axis=1)
        twMat = np.repeat(totWeight, self.R).reshape(self.X.shape[0], self.R)
        MCPR.U[0] = MCPR.U[0] / twMat
        #print(MCPR.U[0])
        #print(self.rawFeatures)
        rawXfile=self.data_dir+'experimentDemo/rawdataX_'+str(experCount)+'.csv'
        rawYfile=self.data_dir+'experimentDemo/rawdataY_'+str(experCount)+'.csv'
        cprXfile=self.data_dir+'experimentDemo/cprdataX_'+str(experCount)+'.csv'
        cprYfile=self.data_dir+'experimentDemo/cprdataY_'+str(experCount)+'.csv'
        np.savetxt(rawXfile,self.rawFeatures)
        np.savetxt(rawYfile,self.Y)
        np.savetxt(cprXfile, MCPR.U[0])
        np.savetxt(cprYfile,self.Y)

        for train, test in self.ttss:
            print "Evaluating Run:{0}".format(run)
            # get the indices for the training tensor
            trainShape = list(self.X.shape)
            trainShape[0] = len(train)
            trainX = tensorSubset(self.X, train, trainShape)
            trainY = self.Y[train]
            ## Evaluate the raw fit using logistic regression

            baseAUC, basePred = predictionTools.getAUC(self.predModel, self.rawFeatures, self.Y, train, test)
            cprAUC, cprPred = predictionTools.getAUC(self.predModel, MCPR.U[0], self.Y, train, test)
            sumBaseAUC+=baseAUC
            sumCprAUC+=cprAUC
            print('base:'+str(baseAUC))
            print('apr:'+str(cprAUC))
            run = run + 1
        print('**************************************')
        print sumBaseAUC/run
        print sumCprAUC/run
        return sumBaseAUC/run,sumCprAUC/run
    def evaluatePredictionAUC_3(self,experCount):
        run = 0
        sumBaseAUC=0.0
        sumCprAUC=0.0

        testCount=4392
        indexC1=np.where(self.X.subs[:,0]<testCount)
        indexC2=np.where(self.X.subs[:,0]>=testCount)
        #print( indexC1)
        subs1=self.X.subs[indexC1]
        subs2=self.X.subs[indexC2]

        subs2[:,0]=subs2[:,0]-testCount
        vals1=self.X.vals[indexC1]
        vals2=self.X.vals[indexC2]
        size1=np.array([testCount,self.X.shape[1],self.X.shape[2]])
        size2=np.array([self.X.shape[0]-testCount,self.X.shape[1],self.X.shape[2]])
        self.Y[self.Y==0]=-1
        Y1=self.Y[:testCount]
        Y2=self.Y[testCount:]
        #print Y1.shape
        trainingX= sptensor.sptensor(subs1, vals1, size1)
        testX= sptensor.sptensor(subs2, vals2, size2)


        MCPR, cpstats, mstats = cp_apr_logis.cp_apr(trainingX, Y1, self.R, maxiters=100, maxinner=50)
        #MCPR, cpstats, mstats = CP_APR.cp_apr(trainingX, self.R, maxiters=100, maxinner=self.innerIter)
        MCPR.normalize_sort(1)

        klproj = KLProjection.KLProjection(MCPR.U, self.R)
        np.random.seed(10)
        testMatrix=klproj.projectSlice(testX, 0)

        ## scale by summing across the rows
        totWeight = np.sum(testMatrix, axis=1)
        zeroIdx = np.where(totWeight < 1e-100)[0]
        if len(zeroIdx) > 0:
            # for the zero ones we're going to evenly distribute
            evenDist = np.repeat(1.0 / self.R, len(zeroIdx)*self.R)
            testMatrix[zeroIdx, :] = evenDist.reshape((len(zeroIdx), self.R))
            totWeight = np.sum(testMatrix, axis=1)
        twMat = np.repeat(totWeight, self.R).reshape(testMatrix.shape[0], self.R)
        testMatrix = testMatrix / twMat
        #print(MCPR.U[0])
        #print(self.rawFeatures)
        rawXfile=self.data_dir+'experiment/trainingX_'+str(experCount)+'.csv'
        rawYfile=self.data_dir+'experiment/trainingY_'+str(experCount)+'.csv'
        cprXfile=self.data_dir+'experiment/testX_'+str(experCount)+'.csv'
        cprYfile=self.data_dir+'experiment/testY_'+str(experCount)+'.csv'
        np.savetxt(rawXfile,MCPR.U[0])
        np.savetxt(rawYfile,Y1)
        np.savetxt(cprXfile, testMatrix)
        np.savetxt(cprYfile,Y2)

        print 'OK'

    def evaluatePredictionAUC_4(self,experCount,trainTensor, trainY,testTensor,testY):
        MCPR, cpstats = cp_apr_logis.cp_apr(trainTensor, trainY, self.R, maxiters=1, maxinner=10)
        MCPR.normalize_sort(0)
        #MCPR.normalize_absorb(0, 2)

        totWeight = np.sum(MCPR.U[0], axis=1)
        zeroIdx = np.where(totWeight < 1e-100)[0]
        if len(zeroIdx) > 0:
            # for the zero ones we're going to evenly distribute
            evenDist = np.repeat(1.0 / self.R, len(zeroIdx)*self.R)
            MCPR.U[0][zeroIdx, :] = evenDist.reshape((len(zeroIdx), self.R))
            totWeight = np.sum(MCPR.U[0], axis=1)
        twMat = np.repeat(totWeight, self.R).reshape(self.X.shape[0], self.R)
        MCPR.U[0] = MCPR.U[0] / twMat

        klproj = KLProjection.KLProjection(MCPR.U, self.R)
        np.random.seed(10)
        testMatrix=klproj.projectSlice(testTensor, 0)

        ## scale by summing across the rows
        totWeight = np.sum(testMatrix, axis=1)
        zeroIdx = np.where(totWeight < 1e-100)[0]
        if len(zeroIdx) > 0:
            # for the zero ones we're going to evenly distribute
            evenDist = np.repeat(1.0 / self.R, len(zeroIdx)*self.R)
            testMatrix[zeroIdx, :] = evenDist.reshape((len(zeroIdx), self.R))
            totWeight = np.sum(testMatrix, axis=1)
        twMat = np.repeat(totWeight, self.R).reshape(testMatrix.shape[0], self.R)
        testMatrix = testMatrix / twMat
        #print(MCPR.U[0])
        #print(self.rawFeatures)
        rawXfile=self.data_dir+'experiment_Data0_corectY_2/trainingX_'+str(experCount)+'.csv'
        rawYfile=self.data_dir+'experiment_Data0_corectY_2/trainingY_'+str(experCount)+'.csv'
        cprXfile=self.data_dir+'experiment_Data0_corectY_2/testX_'+str(experCount)+'.csv'
        cprYfile=self.data_dir+'experiment_Data0_corectY_2/testY_'+str(experCount)+'.csv'

        diagnosisfile=self.data_dir+'experiment_Data0_corectY_2/diagnosis_'+str(experCount)+'.csv'
        medicinefile=self.data_dir+'experiment_Data0_corectY_2/medicine_'+str(experCount)+'.csv'
        np.savetxt(rawXfile,MCPR.U[0])
        np.savetxt(rawYfile,trainY)
        np.savetxt(cprXfile, testMatrix)
        np.savetxt(cprYfile,testY)

        MCPR.normalize_absorb(1, 2)
        np.savetxt(diagnosisfile,MCPR.U[1])
        MCPR.normalize_absorb(2, 2)
        np.savetxt(medicinefile,MCPR.U[2])

        print 'OK'
    def evaluatePredictionAUC_5(self,experCount,trainTensor, trainY,testTensor,testY):

        MCPR, cpstats = cp_apr_linear.cp_apr(trainTensor,  trainY, self.R, maxiters=1, maxinner=10)
        #MCPR, cpstats, mstats = CP_APR.cp_apr(trainingX, self.R, maxiters=100, maxinner=self.innerIter)
        MCPR.normalize_sort(0)
        #MCPR.normalize_absorb(0, 2)

        totWeight = np.sum(MCPR.U[0], axis=1)
        zeroIdx = np.where(totWeight < 1e-100)[0]
        if len(zeroIdx) > 0:
            # for the zero ones we're going to evenly distribute
            evenDist = np.repeat(1.0 / self.R, len(zeroIdx)*self.R)
            MCPR.U[0][zeroIdx, :] = evenDist.reshape((len(zeroIdx), self.R))
            totWeight = np.sum(MCPR.U[0], axis=1)
        twMat = np.repeat(totWeight, self.R).reshape(self.X.shape[0], self.R)
        MCPR.U[0] = MCPR.U[0] / twMat

        klproj = KLProjection.KLProjection(MCPR.U, self.R)
        np.random.seed(10)
        testMatrix=klproj.projectSlice(testTensor, 0)

        ## scale by summing across the rows
        totWeight = np.sum(testMatrix, axis=1)
        zeroIdx = np.where(totWeight < 1e-100)[0]
        if len(zeroIdx) > 0:
            # for the zero ones we're going to evenly distribute
            evenDist = np.repeat(1.0 / self.R, len(zeroIdx)*self.R)
            testMatrix[zeroIdx, :] = evenDist.reshape((len(zeroIdx), self.R))
            totWeight = np.sum(testMatrix, axis=1)
        twMat = np.repeat(totWeight, self.R).reshape(testMatrix.shape[0], self.R)
        testMatrix = testMatrix / twMat
        #print(MCPR.U[0])
        #print(self.rawFeatures)
        rawXfile=self.data_dir+'experiment_Data0_linear_2/trainingX_'+str(experCount)+'.csv'
        rawYfile=self.data_dir+'experiment_Data0_linear_2/trainingY_'+str(experCount)+'.csv'
        cprXfile=self.data_dir+'experiment_Data0_linear_2/testX_'+str(experCount)+'.csv'
        cprYfile=self.data_dir+'experiment_Data0_linear_2/testY_'+str(experCount)+'.csv'

        diagnosisfile=self.data_dir+'experiment_Data0_linear_2/diagnosis_'+str(experCount)+'.csv'
        medicinefile=self.data_dir+'experiment_Data0_linear_2/medicine_'+str(experCount)+'.csv'
        np.savetxt(rawXfile,MCPR.U[0])
        np.savetxt(rawYfile,trainY)
        np.savetxt(cprXfile, testMatrix)
        np.savetxt(cprYfile,testY)

        MCPR.normalize_absorb(1, 2)
        np.savetxt(diagnosisfile,MCPR.U[1])
        MCPR.normalize_absorb(2, 2)
        np.savetxt(medicinefile,MCPR.U[2])

        print 'OK'
def tensorSubset(origTensor, subsetIds, subsetShape):
    """ 
    Get a subset of the tensor specified by the subsetIds
    
    Parameters
    ------------
    X : the original tensor
    subsetIds : a list of indices
    subsetShape : the shape of the new tensor
    
    Output
    -----------
    subsetX : the tensor with the indices rebased
    """
    subsetIdx = np.in1d(origTensor.subs[:,0].ravel(), subsetIds)
    subsIdx = np.where(subsetIdx)[0]
    subsetSubs = origTensor.subs[subsIdx,:]
    subsetVals = origTensor.vals[subsIdx]
    # reindex the 0th mode
    subsetSubs = predictionModel.rebase(subsetIds, subsetSubs)
    return sptensor.sptensor(subsetSubs, subsetVals, subsetShape)