#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:16:20 2019

@author: priyanka
"""

########### ASSIGNMENT 2 ##############
####### PRIYANKA VASANTHAKUMARI ##############


import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import Model
import os
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier

model = VGG16(weights='imagenet',include_top=False)  ## VGG 16 model

os.chdir("/Users/priyanka/Documents/Course works/Pattern Recognition/ECEN649 Project")

## Read the excel sheet

file = pd.read_csv('micrograph.csv')
mic_id = file['micrograph_id']
path = file['path']
label = file['primary_microconstituent']

sph_ind = label[label == 'spheroidite'].index 
len(sph_ind)

net_ind = label[label == 'network'].index 
len(net_ind)

pearl_ind = label[label == 'pearlite'].index 
len(pearl_ind)

os.chdir("/Users/priyanka/Documents/Course works/Pattern Recognition/ECEN649 Project/micrograph")

## Extract features from layers - spherodite

blk1_feat_sph=np.zeros((len(sph_ind),64))
blk2_feat_sph=np.zeros((len(sph_ind),128))
blk3_feat_sph=np.zeros((len(sph_ind),256))
blk4_feat_sph=np.zeros((len(sph_ind),512))
blk5_feat_sph=np.zeros((len(sph_ind),512))



for i in range(0,len(sph_ind)):
    
    image = load_img(path[sph_ind[i]])   #Load
    image = img_to_array(image)         # Convert to array
    image = image[0:484,:,:]            # crop suptitles
    image = np.resize(image,(224,224,3)) # resize
    image = image.reshape(1,image.shape[0],image.shape[1], image.shape[2]) # Add one dimension
    image = preprocess_input(image)  #preprocess the image
    

    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block1_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk1_feat_sph[i,:] = n
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block2_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk2_feat_sph[i,:] = n
    
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block3_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk3_feat_sph[i,:] = n
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block4_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk4_feat_sph[i,:] = n
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk5_feat_sph[i,:] = n

## Extract features from layers - Network

blk1_feat_net=np.zeros((len(net_ind),64))
blk2_feat_net=np.zeros((len(net_ind),128))
blk3_feat_net=np.zeros((len(net_ind),256))
blk4_feat_net=np.zeros((len(net_ind),512))
blk5_feat_net=np.zeros((len(net_ind),512))



for i in range(0,len(net_ind)):
    
    
    image = load_img(path[net_ind[i]])   #Load
    image = img_to_array(image)         # Convert to array
    image = image[0:484,:,:]            # Crop suptitles
    image = np.resize(image,(224,224,3))
    image = image.reshape(1,image.shape[0],image.shape[1], image.shape[2])  # Add dimension
    image = preprocess_input(image)  #preprocess
    
    
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block1_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk1_feat_net[i,:] = n
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block2_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk2_feat_net[i,:] = n
    
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block3_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk3_feat_net[i,:] = n
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block4_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk4_feat_net[i,:] = n
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk5_feat_net[i,:] = n
    

## Extract features from layers - Pearlite

blk1_feat_pearl=np.zeros((len(pearl_ind),64))
blk2_feat_pearl=np.zeros((len(pearl_ind),128))
blk3_feat_pearl=np.zeros((len(pearl_ind),256))
blk4_feat_pearl=np.zeros((len(pearl_ind),512))
blk5_feat_pearl=np.zeros((len(pearl_ind),512))



for i in range(0,len(pearl_ind)):
    
    image = load_img(path[pearl_ind[i]])   #Load
    image = img_to_array(image)         # Convert to array
    image = image[0:484,:,:]            # Crop suptitles
    image = np.resize(image,(224,224,3))
    image = image.reshape(1,image.shape[0],image.shape[1], image.shape[2])
    image = preprocess_input(image)  #preprocess
    
    
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block1_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk1_feat_pearl[i,:] = n
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block2_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk2_feat_pearl[i,:] = n
    
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block3_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk3_feat_pearl[i,:] = n
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block4_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk4_feat_pearl[i,:] = n
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)
    feat = model_extractfeatures.predict(image)
    m = np.mean(feat,axis=1)
    n=np.mean(m,axis=1)
    blk5_feat_pearl[i,:] = n

#### Training data #######
    
#spherodite
    Tr_sph_blk1 =  blk1_feat_sph[0:100,:] 
    Tr_sph_blk2 =  blk2_feat_sph[0:100,:] 
    Tr_sph_blk3 =  blk3_feat_sph[0:100,:] 
    Tr_sph_blk4 =  blk4_feat_sph[0:100,:] 
    Tr_sph_blk5 =  blk5_feat_sph[0:100,:] 

#network
    Tr_net_blk1 =  blk1_feat_net[0:100,:] 
    Tr_net_blk2 =  blk2_feat_net[0:100,:] 
    Tr_net_blk3 =  blk3_feat_net[0:100,:] 
    Tr_net_blk4 =  blk4_feat_net[0:100,:] 
    Tr_net_blk5 =  blk5_feat_net[0:100,:] 

#Pearlite
    Tr_pearl_blk1 =  blk1_feat_pearl[0:100,:] 
    Tr_pearl_blk2 =  blk2_feat_pearl[0:100,:] 
    Tr_pearl_blk3 =  blk3_feat_pearl[0:100,:] 
    Tr_pearl_blk4 =  blk4_feat_pearl[0:100,:] 
    Tr_pearl_blk5 =  blk5_feat_pearl[0:100,:] 
    

####  Test data  ########
    
#spherodite
    Ts_sph_blk1 =  blk1_feat_sph[100:,:] 
    Ts_sph_blk2 =  blk2_feat_sph[100:,:] 
    Ts_sph_blk3 =  blk3_feat_sph[100:,:] 
    Ts_sph_blk4 =  blk4_feat_sph[100:,:] 
    Ts_sph_blk5 =  blk5_feat_sph[100:,:] 

#network
    Ts_net_blk1 =  blk1_feat_net[100:,:] 
    Ts_net_blk2 =  blk2_feat_net[100:,:] 
    Ts_net_blk3 =  blk3_feat_net[100:,:] 
    Ts_net_blk4 =  blk4_feat_net[100:,:] 
    Ts_net_blk5 =  blk5_feat_net[100:,:] 

#Pearlite
    Ts_pearl_blk1 =  blk1_feat_pearl[100:,:] 
    Ts_pearl_blk2 =  blk2_feat_pearl[100:,:] 
    Ts_pearl_blk3 =  blk3_feat_pearl[100:,:] 
    Ts_pearl_blk4 =  blk4_feat_pearl[100:,:] 
    Ts_pearl_blk5 =  blk5_feat_pearl[100:,:] 

### Y-label for train and test sets     ######   
    
Y_Tr_s = np.repeat('S', 100)
Y_Tr_n = np.repeat('N', 100)
Y_Tr_p = np.repeat('P', 100)

Y_Ts_s = np.repeat('S', 274)
Y_Ts_n = np.repeat('N', 112)
Y_Ts_p = np.repeat('P', 24)

##Pairwise classifiers for cossvalidation###

## CV error = 1 - CV score

#SVM1 - Spherodite vs Network - SVM RBF cost 10
    

Tr_1_sn = np.append(Tr_sph_blk1,Tr_net_blk1,axis=0)
Tr_2_sn = np.append(Tr_sph_blk2,Tr_net_blk2,axis=0)
Tr_3_sn = np.append(Tr_sph_blk3,Tr_net_blk3,axis=0)
Tr_4_sn = np.append(Tr_sph_blk4,Tr_net_blk4,axis=0)
Tr_5_sn = np.append(Tr_sph_blk5,Tr_net_blk5,axis=0)
Y_Tr_SN = np.append(Y_Tr_s,Y_Tr_n , axis = 0)

svc_r = SVC(C=10,kernel='rbf')
CV_score_SN = []           ## CV error = 1 - CV score
CV_score_SN.append(np.mean(cross_val_score(svc_r, Tr_1_sn, Y_Tr_SN, cv=10)))
CV_score_SN.append(np.mean(cross_val_score(svc_r, Tr_2_sn, Y_Tr_SN, cv=10)))
CV_score_SN.append(np.mean(cross_val_score(svc_r, Tr_3_sn, Y_Tr_SN, cv=10)))
CV_score_SN.append(np.mean(cross_val_score(svc_r, Tr_4_sn, Y_Tr_SN, cv=10)))
CV_score_SN.append(np.mean(cross_val_score(svc_r, Tr_5_sn, Y_Tr_SN, cv=10)))
max_ind_SN = np.argmax(CV_score_SN)  ## Index of the maximum CV score


#SVM2 - Spherodite vs Pearlite - SVM RBF cost 10
    
Tr_1_sp = np.append(Tr_sph_blk1,Tr_pearl_blk1,axis=0)
Tr_2_sp = np.append(Tr_sph_blk2,Tr_pearl_blk2,axis=0)
Tr_3_sp = np.append(Tr_sph_blk3,Tr_pearl_blk3,axis=0)
Tr_4_sp = np.append(Tr_sph_blk4,Tr_pearl_blk4,axis=0)
Tr_5_sp = np.append(Tr_sph_blk5,Tr_pearl_blk5,axis=0)
Y_Tr_SP = np.append(Y_Tr_s,Y_Tr_p , axis = 0)

CV_score_SP = []           ## CV error = 1 - CV score
CV_score_SP.append(np.mean(cross_val_score(svc_r, Tr_1_sp, Y_Tr_SP, cv=10)))
CV_score_SP.append(np.mean(cross_val_score(svc_r, Tr_2_sp, Y_Tr_SP, cv=10)))
CV_score_SP.append(np.mean(cross_val_score(svc_r, Tr_3_sp, Y_Tr_SP, cv=10)))
CV_score_SP.append(np.mean(cross_val_score(svc_r, Tr_4_sp, Y_Tr_SP, cv=10)))
CV_score_SP.append(np.mean(cross_val_score(svc_r, Tr_5_sp, Y_Tr_SP, cv=10)))
max_ind_SP = np.argmax(CV_score_SP) ## Index of the maximum CV score

#SVM3 - Network vs Pearlite - SVM RBF cost 10
    
Tr_1_np = np.append(Tr_net_blk1,Tr_pearl_blk1,axis=0)
Tr_2_np = np.append(Tr_net_blk2,Tr_pearl_blk2,axis=0)
Tr_3_np = np.append(Tr_net_blk3,Tr_pearl_blk3,axis=0)
Tr_4_np = np.append(Tr_net_blk4,Tr_pearl_blk4,axis=0)
Tr_5_np = np.append(Tr_net_blk5,Tr_pearl_blk5,axis=0)
Y_Tr_NP = np.append(Y_Tr_n,Y_Tr_p , axis = 0)

svc_r = SVC(C=10,kernel='rbf')
CV_score_NP = []           ## CV error = 1 - CV score
CV_score_NP.append(np.mean(cross_val_score(svc_r, Tr_1_np, Y_Tr_NP, cv=10)))
CV_score_NP.append(np.mean(cross_val_score(svc_r, Tr_2_np, Y_Tr_NP, cv=10)))
CV_score_NP.append(np.mean(cross_val_score(svc_r, Tr_3_np, Y_Tr_NP, cv=10)))
CV_score_NP.append(np.mean(cross_val_score(svc_r, Tr_4_np, Y_Tr_NP, cv=10)))
CV_score_NP.append(np.mean(cross_val_score(svc_r, Tr_5_np, Y_Tr_NP, cv=10)))
max_ind_NP = np.argmax(CV_score_NP)  ## Index of the maximum CV score

### Multiple label classifier ## - one vs one

Tr_1_spn = np.append(Tr_1_sp,Tr_net_blk1,axis=0)
Tr_2_spn = np.append(Tr_2_sp,Tr_net_blk2,axis=0)
Tr_3_spn = np.append(Tr_3_sp,Tr_net_blk3,axis=0)
Tr_4_spn = np.append(Tr_4_sp,Tr_net_blk4,axis=0)
Tr_5_spn = np.append(Tr_5_sp,Tr_net_blk5,axis=0)
Y_Tr_SPN = np.append(Y_Tr_SP,Y_Tr_n , axis = 0)

ovo_clf = OneVsOneClassifier(SVC(C=10,kernel='rbf'))
ovo_clf.fit(Tr_1_spn, Y_Tr_SPN)

CV_score_SPN = []           ## CV error = 1 - CV score
CV_score_SPN.append(np.mean(cross_val_score(ovo_clf, Tr_1_spn, Y_Tr_SPN, cv=10)))
CV_score_SPN.append(np.mean(cross_val_score(ovo_clf, Tr_2_spn, Y_Tr_SPN, cv=10)))
CV_score_SPN.append(np.mean(cross_val_score(ovo_clf, Tr_3_spn, Y_Tr_SPN, cv=10)))
CV_score_SPN.append(np.mean(cross_val_score(ovo_clf, Tr_4_spn, Y_Tr_SPN, cv=10)))
CV_score_SPN.append(np.mean(cross_val_score(ovo_clf, Tr_5_spn, Y_Tr_SPN, cv=10)))
max_ind_SPN = np.argmax(CV_score_SPN)  ## Index of the maximum CV score


##### Best feature with minimum CV error - convolution "LAYER 5"  ###

###### TEST ERROR RATES #########

#SVM1 - Spherodite vs Pearlite - SVM RBF cost 10

Tr_5_sp = np.append(Tr_sph_blk5,Tr_pearl_blk5,axis=0)
Y_Tr_SP = np.append(Y_Tr_s,Y_Tr_p , axis = 0)

svc_r.fit(Tr_5_sp, Y_Tr_SP)
y_pred = svc_r.predict(Ts_sph_blk5)    ## Testing with spherodite testing data
accuracy = np.mean(y_pred==Y_Ts_s)
error_rate = 1-accuracy

#SVM2 - Spherodite vs Network - SVM RBF cost 10

Tr_5_sn = np.append(Tr_sph_blk5,Tr_net_blk5,axis=0)
Y_Tr_SN = np.append(Y_Tr_s,Y_Tr_n , axis = 0)

svc_r.fit(Tr_5_sn, Y_Tr_SN)
y_pred = svc_r.predict(Ts_sph_blk5)   ## Testing with spherodite testing data
accuracy = np.mean(y_pred==Y_Ts_s)
error_rate = 1-accuracy


#SVM3 - Network vs Pearlite - SVM RBF cost 10

Tr_5_pn = np.append(Tr_pearl_blk5,Tr_net_blk5,axis=0)
Y_Tr_PN = np.append(Y_Tr_p,Y_Tr_n , axis = 0)

svc_r.fit(Tr_5_pn, Y_Tr_PN)
y_pred = svc_r.predict(Ts_net_blk5)   ## Testing with network testing data
accuracy = np.mean(y_pred==Y_Ts_n)
error_rate = 1-accuracy

## Multilabel classifier

ovo_clf = OneVsOneClassifier(SVC(C=10,kernel='rbf'))
ovo_clf.fit(Tr_5_spn, Y_Tr_SPN)
y_pred = ovo_clf.predict(Ts_net_blk5)   ## Testing with network testing data
accuracy = np.mean(y_pred==Y_Ts_n)
error_rate = 1-accuracy

###############################################################
###############################################################
###############################################################


