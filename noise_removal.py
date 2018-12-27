# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:10:14 2018

@author: Tahlia
"""

import os
import pandas as pd
from data_config import data_config as dc
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
os.chdir(data_path)
df = pd.read_csv('attribute_list.csv',header = 1)

"""
NOISE REMOVAL

This section uses the remove_noise method from the data_config script which uses
face detection to determine if the image has a face or not. If 0 faces are detected, 
the image is removed from the dataset
"""
#Remove any files that are considered noisy i.e. the landscape photos
print('Removing noise...')
df_labels = dc.remove_noise(data_path,df)

"""
SPLITTING THE DATA (Binary classes)

We first start with the binary classes - these were all double checked that they
have only 2 classes using the pd.Dataframe['column_name'].unique() method.

Using the train_test_split method from the sklearn library, we split the data 
differently for each method as the 'stratify' option has been used - this 
means that the data is split while preserving the percentage of samples
for each class

Once split, the generated arrays are saved into csv files for later use
"""

#Change directory to store the csv files
path1 = os.path.join(data_path,'train_test_csv')
os.chdir(path1)
binary_label_names = ['eyeglasses','smiling','young','human']
classes = [1,-1]
x = df_labels['file_name']

for label in binary_label_names:
    y = df_labels[label]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42,stratify=y)
    train = pd.concat((x_train,y_train),axis=1)
    test = pd.concat((x_test,y_test),axis=1)
    train.to_csv((label+'_train.csv'))
    test.to_csv((label+'_test.csv'))
    
"""
CONVERSION OF IMAGES

This part converts the images to different types of data:
- feature locations
- pixel counts
- RGB values for each pixel
- grayscale values for each pixel

Using the data_config scirpt, the images are converted to data which are then
saved as '.npy' files. This saves time later on when implementing models

"""
path2 = os.path.join(data_path,'npy_files')

for label in binary_label_names:
    os.chdir(path1)
    train = pd.read_csv((label+'_train.csv'),header=0,index_col=0)
    train_label = to_categorical(train[label])
    test = pd.read_csv((label+'_test.csv'),header=0,index_col=0)
    test_label = to_categorical(test[label])
    
    ####
    print("Converting training data for greyscale images...")
    train_data_gray = dc.image_to_data_gray(data_path,train)
    print("Converting training data for RGB images...")
    train_data_rgb = dc.image_to_data_rgb(data_path,train)
    print("Converting training data for pixel counts...")
    train_data_pixel = dc.pixel_counts(data_path,train)
    print("Converting training data for feature extractions...")
    train_data_feature = dc.facial_landmark_values(data_path,train)
    ####
    
    ####
    print("Converting testing data for greyscale images...")
    test_data_gray = dc.image_to_data_gray(data_path,test)
    print("Conveerting testing data for RGB images...")
    test_data_rgb = dc.image_to_data_gray(data_path,test)
    print("Converting testing data for pixel counts...")
    test_data_pixel = dc.pixel_counts(data_path,test)
    print("Converting testing data for feature extractions...")
    test_data_feature = dc.facial_landmark_values(data_path,test)
    ####
    os.chdir(path2)
    print("Saving data to .npy files...")
    np.save((str(label)+'_train_gray'),train_data_gray)
    np.save((str(label)+'_train_rgb'),train_data_rgb)
    np.save((str(label)+'_train_pixel'),train_data_pixel)
    np.save((str(label)+'_train_feature'),train_data_feature)
    
    np.save((str(label)+'_test_gray'),test_data_gray)
    np.save((str(label)+'_test_rgb'),test_data_rgb)
    np.save((str(label)+'_test_pixel'),test_data_pixel)
    np.save((str(label)+'_test_feature'),test_data_feature)

    np.save((str(label)+'_train_label'),train_label)
    np.save((str(label)+'_test_label'),test_label)
    
