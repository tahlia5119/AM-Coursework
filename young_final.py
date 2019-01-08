# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 00:50:46 2019

@author: Tahlia
"""

import os
# Ensure the current working directory is in the right place to find
# the lab2_landmarks and the data_config scripts
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
dataset_folder = 'testing_dataset'
script_path = os.path.join(data_path,'scripts')
model_path = os.path.join(data_path,'models_scalers')

os.chdir(script_path)

import pandas as pd
import lab2_landmarks as l2
import numpy as np
np.random.seed(0)
from sklearn.externals import joblib
from data_config import data_config as dc
from sklearn.decomposition import PCA

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

print("Converting training data for RGB data...")
x_test = dc.image_to_data_rgb(os.path.join(data_path,dataset_folder),names,None)

# Load PCA fits
pca_r = joblib.load(os.path.join(data_path,'models_scalers','young_r_pca.pkl'))
pca_g = joblib.load(os.path.join(data_path,'models_scalers','young_g_pca.pkl'))
pca_b = joblib.load(os.path.join(data_path,'models_scalers','young_b_pca.pkl'))

print('Performing PCA...')

x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2],x_test.shape[3])

new_red = pca_r.transform(x_test[:,:,0])
new_green = pca_g.transform(x_test[:,:,1])
new_blue = pca_b.transform(x_test[:,:,2])

# Concatenate the newly reduced datasets together 
print('Concatenating new data...')
x_test = np.concatenate((new_red,new_green,new_blue),axis=1)
    

# Load scaler
scaler = joblib.load(os.path.join(data_path,'models_scalers','young_scaler.pkl'))

# Load the model
model = joblib.load(os.path.join(data_path,'models_scalers','young_model.pkl'))

# Make predictions on the test data
predictions = model.predict(scaler.transform(x_test))

#Save to a csv file
csv_file = pd.DataFrame({'id':names['file_name'],'pred':predictions})

acc = 52.1

csv_file.to_csv(os.path.join(data_path,'test_results','task_2.csv'),header=[acc,''],index=False)
