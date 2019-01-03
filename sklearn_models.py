# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 18:55:36 2018

@author: Tahlia
"""

import os
import numpy as np
import pandas as pd
import time
###CLASSIFIERS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

label = 'human' #choice of 'smiling', 'eyeglasses', 'human', 'young', or 'hair_color'
data_type = 'pixel' #choice of 'pixel', 'feature', or 'rgb_pca'
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
script_path = os.path.join(data_path,'scripts')
npy_path = os.path.join(data_path, 'npy_files')
names = ['LR','RF','Lin SVC','RBF SVC','KN','DT', 'MLP']
classifiers = {
        LogisticRegression(random_state=42, multi_class='ovr'),
        RandomForestClassifier(random_state=42),
        SVC(random_state=42,kernel='linear'),
        SVC(random_state=42),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=42),
        MLPClassifier(random_state=42)
        }


os.chdir(data_path)
train_label = pd.read_csv(os.path.join(data_path,'train_test_csv',label+'_train.csv'),header = 0,index_col=0)
test_label = pd.read_csv(os.path.join(data_path,'train_test_csv',label+'_test.csv'),header = 0,index_col=0)

print('Getting data...')
if data_type != 'rgb_pca':
    train_data = np.load(os.path.join(npy_path,label+'_train_'+data_type+'.npy'))[:,:-1]
    test_data = np.load(os.path.join(npy_path,label+'_test_'+data_type+'.npy'))[:,:-1]
else:
    train_data = np.load(os.path.join(data_path,'npy_pca_files',label+'_train_rgb.npy'))[:,:-1]
    test_data = np.load(os.path.join(data_path,'npy_pca_files',label+'_test_rgb.npy'))[:,:-1]
        
scaler = StandardScaler()

train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

for name,clf in zip(names,classifiers):
    print()
    print(name)
    start = time.time()
    clf.fit(train_data,train_label[label])
    run = time.time()-start
    score = clf.score(test_data,test_label[label])
    print('Accuracy: ',round(score,2)*100,'%')
    print('Fitting time: ',round(run,2),' seconds')
    print()