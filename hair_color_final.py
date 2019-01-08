# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 01:16:34 2019

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
from keras import optimizers
from keras.models import model_from_json

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

# Load encoder
encoder = joblib.load(os.path.join(data_path,'models_scalers','hair_color_encoder.pkl'))

# Load sgd optimizer
sgd = optimizers.SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True)

# load json and create model
json_file = open(os.path.join(data_path,'models_scalers','hair_color_model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join(data_path,'models_scalers','best_model_hair_color.h5'))
print("Loaded model from disk")

# Make predictions on the test data
predictions_oh = loaded_model.predict(x_test)
predictions = encoder.inverse_transform(predictions_oh)

#Save to a csv file
csv_file = pd.DataFrame({'id':names['file_name'],'pred':predictions[:,0]})

acc = 91.9

csv_file.to_csv(os.path.join(data_path,'test_results','task_5.csv'),header=[acc,''],index=False)