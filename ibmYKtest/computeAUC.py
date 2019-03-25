__author__ = 'smileyk01'

import numpy as np

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn import tree


# Loading the Digits dataset
print('loading data...')
data_dir = 'E:/test_project_python/XiamenData/experiment_Data0_corectY_2'
#rawX=np.loadtxt(data_dir+'experiment/rawdataX_50phens_fullmat.csv',delimiter=' ',unpack=True,dtype='double')
#rawX=rawX.transpose()

cprX=np.loadtxt(data_dir+'/testX_50phens_Uapr_100outIter50InIter_lalaa_correct_C10.csv',delimiter=' ',unpack=True,dtype='double')
cprX=cprX.transpose()
Y=np.loadtxt(data_dir+'/testY_50phens_Uapr_100outIter50InIter_lalaa_correct_C10.csv',delimiter=' ',unpack=True,dtype='double')
cprX[np.isnan(cprX) == True] = 0

ttX=np.loadtxt(data_dir+'/trainingX_50phens_Uapr_100outIter50InIter_lalaa_correct_C10.csv',delimiter=' ',unpack=True,dtype='double')
ttX=ttX.transpose()
ttY=np.loadtxt(data_dir+'/trainingY_50phens_Uapr_100outIter50InIter_lalaa_correct_C10.csv',delimiter=' ',unpack=True,dtype='double')
ttX[np.isnan(ttX) == True] = 0
print ttX.shape
# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:


# Split the dataset in two equal parts
#X_train, X_test, y_train, y_test = train_test_split(
    #rawX, Y, test_size=0.5, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(
    cprX, Y, test_size=0.5, random_state=10)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3, 1e-4],
                     'C': [1,10,100,1000,10000,990000]},
                    {'kernel': ['linear'], 'C': [1,10,100,1000,10000,990000]}]
#tuned_parameters = { 'tol':[1e-3, 1e-4], 'C': [1, 10, 100, 1000]}


scores = ['roc_auc']
"""
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(SVC(), tuned_parameters, cv=20,
                       scoring=score)

    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('auc:'+str(auc))
    print('********************************************************************************')
"""
def normX(tX):
    sumttX=np.sum(tX,1).reshape((tX.shape[0],1))
    for i in range(tX.shape[0]):
        if sumttX[i]<=0:
            tX[i]=np.repeat([0.02],tX.shape[1],axis=0)
        else:
            tX[i]=ttX[i]/np.repeat(sumttX[i],tX.shape[1],axis=0)
    return tX
model=LogisticRegression(penalty='l2',C=0.01)
#model=tree.DecisionTreeClassifier()
ttY[ttY==-1]=0

#ttX=normX(ttX)
#cprX=normX(cprX)
#print ttX[:10]
#print cprX[:10]
#print np.count_nonzero(ttY)
#ttX=np.hstack((indexttX,ttX))
#print ttX[0:10,:]
model.fit(ttX, ttY)
result = model.predict_proba(cprX)[:,1]
Y[Y==-1]=0
print(result)
print Y
fpr, tpr, thresholds = metrics.roc_curve(Y, result, pos_label=1)
auc = metrics.auc(fpr, tpr)
print('auc:'+str(auc))

