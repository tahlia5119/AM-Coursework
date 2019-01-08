# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 00:22:26 2019

@author: Tahlia
"""

import os
# Ensure the current working directory is in the right place to find
# the lab2_landmarks and the data_config scripts
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
dataset_folder = 'testing_dataset'
script_path = os.path.join(data_path,'scripts')
os.chdir(script_path)

import pandas as pd
from data_config import data_config as dc
import lab2_landmarks as l2
import numpy as np
np.random.seed(0)
from sklearn.externals import joblib

"""
NOISE REMOVAL

This section uses the extract_features method from the lab2_landmarks script
 which uses face detection to determine if the image has a face or not. 
If 0 faces are detected, the image is removed from the dataset

It was noted that the labels contain negative numbers in them which makes it difficult
using the above libraries to convert the labels to one hot labels. Hence, before the
data processing, the labels are changed so that they are positive integers or 0.
"""
# Remove any files that are considered noisy i.e. the landscape photos
print('Removing noise...')

#Create dataframe of the filenames without the file extension
os.chdir(os.path.join(data_path,dataset_folder))
files = os.listdir()
filenames = [int(f[:-4]) for f in files]
filenames.sort()
names = pd.DataFrame(filenames, columns = ['file_name'])

print("Converting training data for feature extractions...")
file_id, x_test = l2.extract_features_labels(os.path.join(data_path,dataset_folder),names)

# Load scaler
scaler = joblib.load(os.path.join(data_path,'models_scalers','smiling_scaler.pkl'))

# Load the model
model = joblib.load(os.path.join(data_path,'models_scalers','smiling_model.pkl'))

# Make predictions on the test data
predictions = model.predict(scaler.transform(x_test.drop('file_name',axis=1)))

#Save to a csv file
csv_file = pd.DataFrame({'id':file_id['file_name'],'pred':predictions})

acc = 92.8

csv_file.to_csv(os.path.join(data_path,'test_results','task_1.csv'),header=[acc,''],index=False)
