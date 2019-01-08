# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:10:14 2018

@author: Tahlia
"""

import os
# Ensure the current working directory is in the right place to find
# the lab2_landmarks and the data_config scripts
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
dataset = os.path.join(data_path,'dataset')
script_path = os.path.join(data_path,'scripts')
os.chdir(script_path)

import pandas as pd
from data_config import data_config as dc
import lab2_landmarks as l2
import numpy as np
np.random.seed(0)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical

# Change back to the main folder path
os.chdir(data_path)
#Read the attribute list from CSV into pandas DataFrame
df = pd.read_csv('attribute_list.csv',header = 1)

# I want to keep the images the same size to maintain as much information
# as about the image
target_size = None

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

# Return the attribute list of the remaining images and their respective
# facial landmark values
df_labels,landmark_features = l2.extract_features_labels(dataset,df)
landmark_features.index = landmark_features['file_name']-1
binary_labels = ['eyeglasses','smiling','young','human']

# Change any -1 values to 0 so that the binary tasks can be one hot 
# encoded later on
print('Changing labels...')
for label in binary_labels:
    df_labels.loc[df_labels[label]==-1, label] = 0

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

print('Splitting the data...')

# Change directory to store the csv files
path1 = os.path.join(data_path,'train_test_csv')
os.chdir(path1)

x = landmark_features

# Initialize the encoder
encoder = OneHotEncoder(sparse=False, categories = 'auto')

for label in binary_labels:
    # Get the attribute for the specific task
    y = df_labels[label]
    
    # Split the data into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1,random_state=42,stratify=y)
    
    # Further split the data into train and validation datasets
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=42, stratify=y_train)
    
    # Concatenate the different arrays together respectively to avoid any
    # risk of attributes not properly aligned with the previous index
    train = pd.concat((x_train,y_train),axis=1)
    val = pd.concat((x_val,y_val),axis=1)
    test = pd.concat((x_test,y_test),axis=1)
    
    # Save all of the new label files as CSVs
    train.to_csv((label+'_train.csv'),index=False)
    val.to_csv((label+'_val.csv'),index=False)
    test.to_csv((label+'_test.csv'),index=False)
    
"""
CONVERSION OF IMAGES (Binary classes)

This part converts the images to different types of data:
- facial landmark coordinates
- pixel counts
- RGB values for each pixel
- grayscale values for each pixel

Using the data_config script, the images are converted to data which are then
saved as '.npy' files. This saves time later on when implementing models

"""
path2 = os.path.join(data_path,'npy_files')

for label in binary_labels:
    
    # Change to the CSV file location
    os.chdir(path1)
    
    # Read the training data label into Dataframe
    train = pd.read_csv((label+'_train.csv'),header=0,index_col=False)
    
    # Reset the index to reflect the original index before splitting
    train.index = train['file_name']-1
    
    # Fit the encoder to the training label and transform it to a 
    # one hot label 
    train_label = encoder.fit_transform(np.array(train[label]).reshape(-1,1))
    
    # Read the testing data label into Dataframe
    test = pd.read_csv((label+'_test.csv'),header=0,index_col=False)
    
    # Reset the index
    test.index = test['file_name']-1
    
    # Use the fitted encoder to transform the testing label to a 
    # one hot label
    test_label = encoder.transform(np.array(test[label]).reshape(-1,1))
    
    # Read the validation label into Dataframe
    val = pd.read_csv((label+'_val.csv'),header=0,index_col=False)
    
    #Reset the index
    val.index = val['file_name']-1
    
    # Use the fitted encoder to transform the validation label to
    # a one hot label
    val_label = encoder.transform(np.array(val[label]).reshape(-1,1)) 
    
    ####### DATA CONVERSION #######
    
    ####
    print("Converting training data for greyscale images...")
    train_data_gray = dc.image_to_data_gray(dataset,train,target_size)
    print("Converting training data for RGB images...")
    train_data_rgb = dc.image_to_data_rgb(dataset,train,target_size)
    print("Converting training data for pixel counts...")
    train_data_pixel = dc.pixel_counts(dataset,train)
    print("Converting training data for feature extractions...")
    train_data_feature = np.array(train.drop(['file_name',label],axis=1))
    ####
    
    ####
    print("Converting validation data for greyscale images...")
    val_data_gray = dc.image_to_data_gray(dataset,val,target_size)
    print("Conveerting validation data for RGB images...")
    val_data_rgb = dc.image_to_data_rgb(dataset,val,target_size)
    print("Converting validation data for pixel counts...")
    val_data_pixel = dc.pixel_counts(dataset,val)
    print("Converting validation data for feature extractions...")
    val_data_feature = val.drop(['file_name',label],axis=1)
    ####
    
    ####
    print("Converting testing data for greyscale images...")
    test_data_gray = dc.image_to_data_gray(dataset,test,target_size)
    print("Conveerting testing data for RGB images...")
    test_data_rgb = dc.image_to_data_rgb(dataset,test,target_size)
    print("Converting testing data for pixel counts...")
    test_data_pixel = dc.pixel_counts(dataset,test)
    print("Converting testing data for feature extractions...")
    test_data_feature = test.drop(['file_name',label],axis=1)
    ####
    
    # Change to the npy_files folder
    os.chdir(path2)
    
    #Save the newly created arrays to npy files for later use
    print("Saving data to .npy files...")
    np.save((label+'_train_gray'),train_data_gray)
    np.save((label+'_train_rgb'),train_data_rgb)
    np.save((label+'_train_pixel'),train_data_pixel)
    np.save((label+'_train_feature'),train_data_feature)
    
    np.save((label+'_test_gray'),test_data_gray)
    np.save((label+'_test_rgb'),test_data_rgb)
    np.save((label+'_test_pixel'),test_data_pixel)
    np.save((label+'_test_feature'),test_data_feature)
    
    np.save((label+'_val_gray'),val_data_gray)
    np.save((label+'_val_rgb'),val_data_rgb)
    np.save((label+'_val_pixel'),val_data_pixel)
    np.save((label+'_val_feature'),val_data_feature)    

    np.save((label+'_train_label'),train_label)
    np.save((label+'_val_label'),val_label)
    np.save((label+'_test_label'),test_label)

   
"""
SPLITTING THE DATA (Multiclass)

I first double checked the number of unique classes (7; -1,0,1,2,3,4,5) in the hair_color label 
and found that the number of classes did not correspond to the number of different
hair colors (6; bald, blonde, ginger, brown, black,grey).
"""

# Change to the CSV folder
os.chdir(path1)

# Drop any rows where hair_color=-1
df_labels = df_labels[df_labels.hair_color != -1]

# Create an array of the file names from the resulting label dataframe
# and then drop rows from landmark_features that correspond to the
# image sthat have hair_color=-1
fid = np.array(df_labels['file_name'])
x=landmark_features[landmark_features.file_name.isin(fid)]
y = df_labels['hair_color']

# Split the data into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1,random_state=42,stratify=y)

# Further split the data into train and validation datasets
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1,random_state=42,stratify=y_train)

# Concatenate the different arrays together respectively to avoid any
# risk of attributes not properly aligned with the previous index
train = pd.concat((x_train,y_train),axis=1)
val = pd.concat((x_val,y_val),axis=1)
test = pd.concat((x_test,y_test),axis=1)

# Save all of the new label files as CSVs
train.to_csv(('hair_color_train.csv'))
val.to_csv(('hair_color_val.csv'))
test.to_csv(('hair_color_test.csv'))

"""
Converting the data (multiclass)
"""

# Fit the encoder to the training label and transform to a one hot label
train_label = encoder.fit_transform(np.array(train['hair_color']).reshape(-1,1))

# Transform the validation label to a one hot label
val_label = encoder.transform(np.array(val['hair_color']).reshape(-1,1))

# Transform the testing label to a one hot label
test_label = encoder.transform(np.array(test['hair_color']).reshape(-1,1))

####
print("Converting training data for greyscale images...")
train_data_gray = dc.image_to_data_gray(dataset,train,target_size)
print("Converting training data for RGB images...")
train_data_rgb = dc.image_to_data_rgb(dataset,train,target_size)
print("Converting training data for pixel counts...")
train_data_pixel = dc.pixel_counts(dataset,train)
print("Converting training data for feature extractions...")
train_data_feature = train.drop(['file_name','hair_color'],axis=1)
####

####
print("Converting validation data for greyscale images...")
val_data_gray = dc.image_to_data_gray(dataset,val,target_size)
print("Conveerting validation data for RGB images...")
val_data_rgb = dc.image_to_data_rgb(dataset,val,target_size)
print("Converting validation data for pixel counts...")
val_data_pixel = dc.pixel_counts(dataset,val)
print("Converting validation data for feature extractions...")
val_data_feature = val.drop(['file_name','hair_color'],axis=1)
####

####
print("Converting testing data for greyscale images...")
test_data_gray = dc.image_to_data_gray(dataset,test,target_size)
print("Conveerting testing data for RGB images...")
test_data_rgb = dc.image_to_data_rgb(dataset,test,target_size)
print("Converting testing data for pixel counts...")
test_data_pixel = dc.pixel_counts(dataset,test)
print("Converting testing data for feature extractions...")
test_data_feature = test.drop(['file_name','hair_color'],axis=1)
####

# Change to the npy_files folder
os.chdir(path2)

# Save the new arrays for lagter use
print("Saving data to .npy files...")
np.save(('hair_color_train_gray'),train_data_gray)
np.save(('hair_color_train_rgb'),train_data_rgb)
np.save(('hair_color_train_pixel'),train_data_pixel)
np.save(('hair_color_train_feature'),train_data_feature)

np.save(('hair_color_val_gray'),val_data_gray)
np.save(('hair_color_val_rgb'),val_data_rgb)
np.save(('hair_color_val_pixel'),val_data_pixel)
np.save(('hair_color_val_feature'),val_data_feature)

np.save(('hair_color_test_gray'),test_data_gray)
np.save(('hair_color_test_rgb'),test_data_rgb)
np.save(('hair_color_test_pixel'),test_data_pixel)
np.save(('hair_color_test_feature'),test_data_feature)

np.save(('hair_color_train_label'),train_label)
np.save(('hair_color_val_label'),val_label)
np.save(('hair_color_test_label'),test_label)