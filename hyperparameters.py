# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:15:20 2019

@author: Tahlia
"""

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

###Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import time
import numpy as np
import os
import pandas as pd

data_path = 'D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
npy_path = os.path.join(data_path,'npy_files')
npy_pca_path = os.path.join(data_path, 'npy_pca_files')
scaler = StandardScaler()

#Create a dictionary to store the respective classifiers, datatypes, and parameter selections
'''
Label: Smiling
Classifier: MLPClassifier
Data type: Landmarks

Label: Eyeglasses
Classifier: Random Forest
Data type: Pixel counts

Label:Human
Classifier: MLPClassifier
Data type: rgb_pca

Label: Young
Classifier: Linear SVC
Data type: rgb_pca

Label: Hair Color
Classifier: Linear SVC
Data type: rgb_pca
'''

rbf_svc = {'C': [0.1,0.3,1,3,10,30],#36
           'gamma': ['scale',0.001,0.01,0.1,0.3,1],
           'random_state': [42]}
lin_svc = {'kernel': ['linear'],
           'C': [0.1,0.3,1,3,10,30],
           'gamma': ['scale',0.001,0.01,0.1,0.3,1],
           'random_state': [42]}
rf = {'n_estimators':[10,100,200],
      'max_depth': [5,15,None],
      'min_samples_split': [2,3],
      'min_samples_leaf': [1,2],
           'random_state': [42]}
mlp = {'hidden_layer_sizes': [(100,),(100,2,)], 
       'activation': ['relu'],
       'solver': ['sgd','adam'],
       'learning_rate': ['constant','adaptive'],
       'learning_rate_init':[0.001,0.0001],
       'random_state': [42],
       'max_iter': [500]}

whole_dict = {'smiling': {'classifier': MLPClassifier(),
                          'data': 'feature',
                          'params': mlp,
                          'path': npy_path},
              'eyeglasses': {'classifier': RandomForestClassifier(),
                             'data': 'pixel',
                             'params': rf,
                             'path': npy_path},
              'human': {'classifier': MLPClassifier(),
                        'data': 'rgb_pca',
                        'params': mlp,
                        'path': npy_pca_path},
              'young': {'classifier': SVC(),
                        'data': 'rgb_pca',
                        'params': lin_svc,
                        'path': npy_pca_path},
              'hair_color': {'classifier': SVC(),
                             'data': 'rgb_pca',
                             'params': lin_svc,
                             'path': npy_pca_path}
              }

#Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def hyper_param(clf,param_dist,features,label):
    #Building classifier
    #Running grid search
    grid_search = GridSearchCV(clf,param_dist,cv=StratifiedKFold(10),verbose=1)
    start = time.time()
    grid_search.fit(features, label)
    print("GridSearchCV took ",(time.time() - start), " seconds", report(grid_search.cv_results_)," parameter settings.")
    
for label in whole_dict:
    
    print('Label: ',label)
    clf = whole_dict[label]['classifier']
    data = whole_dict[label]['data']
    
    if data == 'rgb_pca':
        data = 'rgb'
        
    param = whole_dict[label]['params']
    path = whole_dict[label]['path']
    
    # Open the label data
    train_label = pd.read_csv(os.path.join(data_path,'train_test_csv',label+'_train.csv'),header = 0,index_col=0)[label]
    val_label = pd.read_csv(os.path.join(data_path,'train_test_csv',label+'_val.csv'),header = 0,index_col=0)[label]
    train_label = pd.concat([train_label,val_label])
    
    # Open the feature data
    train_data = np.load(os.path.join(data_path,path,label+'_train_'+data+'.npy'))
    val_data = np.load(os.path.join(data_path,path,label+'_val_'+data+'.npy'))
    train_data = np.concatenate((train_data,val_data))
    
    # Scale the training data
    train_data = scaler.fit_transform(train_data)
    
    hyper_param(clf,param,train_data,train_label)
  