# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:37:28 2019

@author: Tahlia
"""

from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

# Main folder path
data_path = 'D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'

# Location of numpy arrays
npy_path = os.path.join(data_path,'npy_files')

# Location of numpy arrays on which PCA was performed
npy_pca_path = os.path.join(data_path, 'npy_pca_files')

#File name and loction for the learning curve that is output
graph_path = os.path.join(data_path,'train_curves','mlp_smiling_feature_10_acc.png')

# Initializing scalre
scaler = StandardScaler()

label = 'smiling'
data_type ='feature'

# MLP classifier with selected parameters
clf = MLPClassifier(hidden_layer_sizes=(100,2),solver='sgd',learning_rate='constant', max_iter = 500, learning_rate_init = 0.001, random_state=42)  

# Open labels as dataframes and data as numpy arrays
train_label = pd.read_csv(os.path.join(data_path,'train_test_csv',label+'_train.csv'),header = 0,index_col=0)[label]
train_data = np.load(os.path.join(data_path,npy_path,label+'_train_'+data_type+'.npy'))
#val_label = pd.read_csv(os.path.join(data_path,'train_test_csv',label+'_val.csv'),header = 0,index_col=0)[label]
#val_data = np.load(os.path.join(data_path,npy_path,label+'_val_'+data_type+'.npy'))

# Since cross validation is used, I can combine the train and validation sets to
# have more training smaples
#train_label = pd.concat([train_label,val_label],axis=0)
#train_data = np.concatenate((train_data,val_data))

# Scale the training data
train_data = scaler.fit_transform(train_data)


# Visualize learning curve
train_sizes, train_scores, test_scores = learning_curve(clf, train_data, train_label, scoring='accuracy', cv=10,train_sizes=np.linspace(0.1, 1, 15))

plt.figure()
plt.plot(train_sizes, -test_scores.mean(1), '-o', color="r",label="Test scores")
plt.plot(train_sizes, -train_scores.mean(1), '-o', color="g", label="Train scores")
plt.xlabel("Train size")
plt.ylabel("log loss")
plt.title('Learning curves')
plt.legend(loc="best")
plt.savefig(graph_path)
plt.show()




