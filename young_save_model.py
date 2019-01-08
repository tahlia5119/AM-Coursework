# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:27:25 2019

@author: Tahlia
"""


import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.externals import joblib 

clf = SVC(C=0.1,gamma='scale',random_state=42)

#Main folder path
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'

#location of all the scripts
script_path = os.path.join(data_path,'scripts')

#save the numpy arrays to this folder
npy_path = os.path.join(data_path, 'npy_files')

# save the models and scalers to this folder
model_path = os.path.join(data_path,'models_scalers')

#Change to the main folder
os.chdir(data_path)

#Initialize scaler
scaler = StandardScaler()

train_data = np.load(os.path.join(data_path,'npy_pca_files','young_train_rgb.npy'))
val_data = np.load(os.path.join(data_path,'npy_pca_files','young_val_rgb.npy'))
train_data = np.concatenate((train_data,val_data))

# Fit the scaler to the training data, then transform both the training
# and validation data      
scaler.fit(train_data)
train_data = scaler.transform(train_data)

#Save the scaler
joblib.dump(scaler, os.path.join(model_path,('young_scaler.pkl')))

# Open the files containing the label into a dataframe
train_label = pd.read_csv(os.path.join(data_path,'train_test_csv','young_train.csv'),header = 0,index_col=0)['young']
test_label = pd.read_csv(os.path.join(data_path,'train_test_csv','young_val.csv'),header = 0,index_col=0)['young']
train_label = pd.concat([train_label,test_label])

# Fit the model to the training data
clf.fit(train_data,train_label)

# Save the model
joblib.dump(clf, os.path.join(model_path,('young_model.pkl')))