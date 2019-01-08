# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:56:27 2019

@author: Tahlia
"""

import os
import numpy as np
import pandas as pd

from sklearn.externals import joblib 

# Use balanced_accuracy_score to look at performance since there will be an imbalance of classes
from sklearn.metrics import balanced_accuracy_score

label = 'young'
data_type = 'rgb'

#Main folder path
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'

#location of all the scripts
script_path = os.path.join(data_path,'scripts')

# load the numpy arrays from this folder
npy_path = os.path.join(data_path,'npy_pca_files') 

# Load scaler
scaler = joblib.load(os.path.join(data_path,'models_scalers','young_scaler.pkl'))

# Load the model
model = joblib.load(os.path.join(data_path,'models_scalers','young_model.pkl'))

print('Getting data...')
x_test = np.load(os.path.join(npy_path,label+'_test_'+data_type+'.npy'))
x_test = scaler.transform(x_test)

# Open the files containing the label into a dataframe
y_test = pd.read_csv(os.path.join(data_path,'train_test_csv',label+'_test.csv'),header = 0,index_col=0)[label]

# Make predictions on the test data
predictions = model.predict(x_test)

# Score the test data
score = model.score(x_test,y_test)

# Get balanced accuracy score
balanced = balanced_accuracy_score(y_test,predictions)