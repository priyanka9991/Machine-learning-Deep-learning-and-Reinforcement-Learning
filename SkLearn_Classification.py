#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:31:03 2019

@author: priyanka
"""


########### ASSIGNMENT 1 ##############
####### PRIYANKA VASANTHAKUMARI ##############


import numpy as np
import pandas as pd
import os
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from numpy import unravel_index
from sklearn.svm import SVC




##Load data 
os.chdir("/Users/priyanka/Documents/Course works/Pattern Recognition/ECEN649 Project")
train = pd.read_csv('Assign1_Training_Data.txt',delimiter="\t")
test = pd.read_csv('Assign1_Testing_Data.txt',delimiter="\t")

## Train data
X=train.iloc[:,1:71]
X = X.as_matrix()
Y=train.iloc[:,71]

## Test data
X_test=test.iloc[:,1:71]
X_test = X_test.as_matrix()
Y_test=test.iloc[:,71]

######### Resubstitution error #############
############ ALL Genes - No feature selection ####################

## SVM Linear Cost 1
svc = SVC(C=1,kernel='linear')
svc.fit(X, Y)
svc.score(X, Y)
y_pred=svc.predict(X)
accuracy = np.mean(Y== y_pred)
error_rate = 1 - accuracy              ## Total resubstituion error
class0_err =1-np.mean(Y[Y==0]== y_pred[Y==0])   # Resubstitutin class 0 error
class1_err =1-np.mean(Y[Y==1]== y_pred[Y==1])   # Resubstitutin class 1 error


##SVM RBF cost 10
svc_r = SVC(C=10,kernel='rbf')
svc_r.fit(X, Y)
svc_r.score(X, Y)
y_pred=svc_r.predict(X)
accuracy = np.mean(Y== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y[Y==0]== y_pred[Y==0])
class1_err =1-np.mean(Y[Y==1]== y_pred[Y==1])



## Neural network
clf_nn = MLPClassifier(hidden_layer_sizes=(5,5),activation='logistic',solver='lbfgs',random_state=0)
clf_nn.fit(X, Y)
y_pred = clf_nn.predict(X)
clf_nn.score(X,Y)
accuracy = np.mean(Y== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y[Y==0]== y_pred[Y==0])
class1_err =1-np.mean(Y[Y==1]== y_pred[Y==1])



################# Test Error ###########
###### ALL GENES ##########

##Nofeature selection
## SVM Linear Cost 1
svc = SVC(C=1,kernel='linear')
svc.fit(X, Y)
svc.score(X_test, Y_test)
y_pred=svc.predict(X_test)
accuracy = np.mean(Y_test== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y_test[Y_test==0]== y_pred[Y_test==0])
class1_err =1-np.mean(Y_test[Y_test==1]== y_pred[Y_test==1])


##SVM RBF cost 10
svc_r = SVC(C=10,kernel='rbf')
svc_r.fit(X, Y)
svc_r.score(X_test, Y_test)
y_pred=svc_r.predict(X_test)
accuracy = np.mean(Y_test== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y_test[Y_test==0]== y_pred[Y_test==0])
class1_err =1-np.mean(Y_test[Y_test==1]== y_pred[Y_test==1])



## Neural network
clf_nn = MLPClassifier(hidden_layer_sizes=(5,5),activation='logistic',solver='lbfgs',random_state=0)
clf_nn.fit(X, Y)
clf_nn.score(X_test, Y_test)
y_pred=clf_nn.predict(X_test)
accuracy = np.mean(Y_test== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y_test[Y_test==0]== y_pred[Y_test==0])
class1_err =1-np.mean(Y_test[Y_test==1]== y_pred[Y_test==1])



############ FEATURE SELECTION #####################
    
## Sequential forward search for feature selection ####

X=train.iloc[:,1:71]
n=5;             # Target number of features - change to 3, 4 or 5
ncol = X.shape[1]
feat_set=np.zeros((80,n)) # Contains final feature set
score_max_ind=[]
score_max =[]

for i in range(0,n):
    score=np.zeros(71)
    for j in range (0,ncol):
        
        if j not in score_max_ind:
            X_fs = X.iloc[:,j]
            X_fs = X_fs.as_matrix()
            feat_set[:,i] = X_fs 
            svc.fit(feat_set, Y)
            score[j]= svc.score(feat_set, Y)
    
    max_scoreind = np.argmax(score)
    score_max_ind.append((max_scoreind)) # Contains indices of features selected
    score_max.append(max(score))
    sel = X.iloc[:, max_scoreind]
    sel=sel.as_matrix()
    feat_set[:,i] = sel # final feature set
    
    
## Exhaustive search for feature selection ###
    
n=2 # Target number of features

X_ex=np.zeros((80,2))

score_ex=np.zeros((70,70))
   
for i in range (0,ncol):
    X_ex[:,0] = X.iloc[:,i]
    for j in range (0,ncol):
        if i<j:
            X_ex[:,1] =X.iloc[:,j]
            svc.fit(X_ex, Y)
            score_ex[i,j]= svc.score(X_ex, Y)

max_score_ex = max(map(max, score_ex))
max_score_ind_ex=unravel_index(score_ex.argmax(), score_ex.shape)  ## Contains indices of the selected features
feat_set=np.zeros((80,2))                    ## Final feature set
feat_set[:,0]=X.iloc[:,max_score_ind_ex[0]]
feat_set[:,1]=X.iloc[:,max_score_ind_ex[1]]



#### Classifier after feature selection   #######
#### To calculate resubstitution error #######

## SVM Linear Cost 1
svc = SVC(C=1,kernel='linear')
svc.fit(feat_set, Y)
svc.score(feat_set, Y)
y_pred=svc.predict(feat_set)
accuracy = np.mean(Y== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y[Y==0]== y_pred[Y==0])
class1_err =1-np.mean(Y[Y==1]== y_pred[Y==1])


##SVM RBF cost 10
svc_r = SVC(C=10,kernel='rbf')
svc_r.fit(feat_set, Y)
svc_r.score(feat_set, Y)
y_pred=svc_r.predict(feat_set)
accuracy = np.mean(Y== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y[Y==0]== y_pred[Y==0])
class1_err =1-np.mean(Y[Y==1]== y_pred[Y==1])



## Neural network
clf_nn = MLPClassifier(hidden_layer_sizes=(5,5),activation='logistic',solver='lbfgs',random_state=0)
clf_nn.fit(feat_set, Y)
y_pred = clf_nn.predict(feat_set)
clf_nn.score(feat_set,Y)
accuracy = np.mean(Y== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y[Y==0]== y_pred[Y==0])
class1_err =1-np.mean(Y[Y==1]== y_pred[Y==1])



################# Test Error ###########
## After feature selection #####

# Execute this for SFS #
test_feat=X_test.iloc[:,score_max_ind]
####

#Execute this for exhaustive selection#
test_feat=np.zeros((215,2))
test_feat[:,0]=X_test.iloc[:,max_score_ind_ex[0]]
test_feat[:,1]=X_test.iloc[:,max_score_ind_ex[1]]
####

## SVM Linear Cost 1
svc = SVC(C=1,kernel='linear')
svc.fit(feat_set, Y)
svc.score(test_feat, Y_test)
y_pred=svc.predict(test_feat)
accuracy = np.mean(Y_test== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y_test[Y_test==0]== y_pred[Y_test==0])
class1_err =1-np.mean(Y_test[Y_test==1]== y_pred[Y_test==1])


##SVM RBF cost 10
svc_r = SVC(C=10,kernel='rbf')
svc_r.fit(feat_set, Y)
svc_r.score(test_feat, Y_test)
y_pred=svc_r.predict(test_feat)
accuracy = np.mean(Y_test== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y_test[Y_test==0]== y_pred[Y_test==0])
class1_err =1-np.mean(Y_test[Y_test==1]== y_pred[Y_test==1])



## Neural network
clf_nn = MLPClassifier(hidden_layer_sizes=(5,5),activation='logistic',solver='lbfgs',random_state=0)
clf_nn.fit(feat_set, Y)
clf_nn.score(test_feat, Y_test)
y_pred=clf_nn.predict(test_feat)
accuracy = np.mean(Y_test== y_pred)
error_rate = 1 - accuracy
class0_err =1-np.mean(Y_test[Y_test==0]== y_pred[Y_test==0])
class1_err =1-np.mean(Y_test[Y_test==1]== y_pred[Y_test==1])


###############################################################
###############################################################
###############################################################


















