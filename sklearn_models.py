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

label = 'smiling' #choice of 'smiling', 'eyeglasses', 'human', 'young', or 'hair_color'
data_type = 'feature' #choice of 'feature', 'pixel', 'rgb_pca', or 'pixel_pca'

#Main folder path
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'

#location of all the scripts
script_path = os.path.join(data_path,'scripts')

#save the numpy arrays to this folder
npy_path = os.path.join(data_path, 'npy_files')

# Create an array with the classifier names and another array with
# the classifiers initialized with a random_state=42 for reproducible
# results
names = ['LR','RF','Lin SVC','RBF SVC','KN','DT', 'MLP','XGB']
classifiers = [
        LogisticRegression(random_state=42),
        RandomForestClassifier(random_state=42),
        SVC(random_state=42,kernel='linear'),
        SVC(random_state=42),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=42),
        MLPClassifier(random_state=42)
        ]


#Change to the main folder
os.chdir(data_path)

# Open the files containing the label into a dataframe
train_label = pd.read_csv(os.path.join(data_path,'train_test_csv',label+'_train.csv'),header = 0,index_col=0)
test_label = pd.read_csv(os.path.join(data_path,'train_test_csv',label+'_val.csv'),header = 0,index_col=0)

#Initialize scaler
scaler = StandardScaler()

# Ensure that the correct file path is specified for the data that is in the
# npy_files folder or in the npy_pca_files' folder
print('Getting data...')
if data_type not in ['rgb_pca','pixel_pca']:
    train_data = np.load(os.path.join(npy_path,label+'_train_'+data_type+'.npy'))#[:,:-1]
    test_data = np.load(os.path.join(npy_path,label+'_val_'+data_type+'.npy'))#[:,:-1]
else:
    d_type = data_type.split('_')[0]
    train_data = np.load(os.path.join(data_path,'npy_pca_files',label+'_train_'+d_type+'.npy'))#[:,:-1]
    test_data = np.load(os.path.join(data_path,'npy_pca_files',label+'_val_'+d_type+'.npy'))#[:,:-1]

# Fit the scaler to the training data, then transform both the training
# and validation data      
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

#An array to store the scores output by the classifiers
scores = []

for name,clf in zip(names,classifiers):
    
    print()
    print(name)
    
    # Meausre run time
    start = time.time()
    
    #Fit the model to the training data
    clf.fit(train_data,train_label[label])
    
    run = time.time()-start
    
    # Score the model using the validation data
    score = clf.score(test_data,test_label[label])
    
    # Print the accuracies, append new accuracy to scores
    print('Accuracy: ',round(score,3)*100,'%')
    print('Fitting time: ',round(run,2),' seconds')
    print()
    scores.append(round(score,3))

print(scores)