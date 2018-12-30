# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 18:55:36 2018

@author: Tahlia
"""

import os
import numpy as np
import pandas as pd

###CLASSIFIERS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

names = ['LR','RF','Lin SVC','RBF SVC','KN','DT']
classifiers = {
        LogisticRegression(random_state=42, multi_class='ovr'),
        RandomForestClassifier(random_state=42),
        SVC(random_state=42,kernel='linear'),
        SVC(random_state=42),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=42)
        }

data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
os.chdir(data_path)
train_label = pd.read_csv(os.path.join(data_path,'train_test_csv','hair_color_train.csv'),header = 0,index_col=0)
test_label = pd.read_csv(os.path.join(data_path,'train_test_csv','hair_color_test.csv'),header = 0,index_col=0)

print('Getting data...')
train_data = np.load(os.path.join(data_path,'npy_files','hair_color_train_pixel.npy'))[:,:-1]
test_data = np.load(os.path.join(data_path,'npy_files','hair_color_test_pixel.npy'))[:,:-1]

scaler = StandardScaler()

train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

for name,clf in zip(names,classifiers):
    print()
    print(name)
    clf.fit(train_data,train_label['hair_color'])
    score = clf.score(test_data,test_label['hair_color'])
    print('Accuracy: ',score)
    print()