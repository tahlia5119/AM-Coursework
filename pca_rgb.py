# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:09:39 2019

@author: Tahlia

This script uses principal component analysis to reduce the dimensionality of
the RGB features (samples,256,256,3)
"""

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# Main folder path
data_path = 'D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'

# Location of all the scripts
script_path = os.path.join(data_path,'scripts')

# Location of the npy files 
npy_path = os.path.join(data_path, 'npy_files') 

# Location of where the resultant npy files from pca will be saved
pca_path = os.path.join(data_path, 'npy_pca_files')

# To save the pca fit
model_path = os.path.join(data_path,'models_scalers')

labels = ['young']#['smiling','eyeglasses','human','young','hair_color']

# Number of components to which each RGB channel will be reduced
pca = PCA(n_components = 50)

# Initialize scaler
scaler = StandardScaler()

# Iterate through each classification task
for label in labels:
    
    # Open the RGB npy files
    print('Loading files and reshaping...')
    train = np.load(os.path.join(npy_path,label+'_train_rgb.npy'))
    val = np.load(os.path.join(npy_path,label+'_val_rgb.npy'))
    test = np.load(os.path.join(npy_path,label+'_test_rgb.npy'))
    
    # Reshape the arrays so that they are only 3 dimensions where
    # shape = (samples,256*256,3)
    train_reshape = train.reshape(train.shape[0],np.prod(train.shape[1:-1]),3)
    val_reshape = val.reshape(val.shape[0],np.prod(val.shape[1:-1]),3)
    test_reshape = test.reshape(test.shape[0],np.prod(test.shape[1:-1]),3)
    
    # Separate the different channels (red, green, and blue) into new arrays
    # for each training, validation, and test datasets
    red_train = train_reshape[:,:,0]
    green_train = train_reshape[:,:,1]
    blue_train = train_reshape[:,:,2]
    red_val = val_reshape[:,:,0]
    green_val = val_reshape[:,:,1]
    blue_val = val_reshape[:,:,2]    
    red_test = test_reshape[:,:,0]
    green_test = test_reshape[:,:,1]
    blue_test = test_reshape[:,:,2]
    
    print('Scaling data and performing PCA...')
    
    # Fit the pca to the scaled red training dataset, and transform
    # Transform the validation and test datasets using the fitted pca 
    new_red_train = pca.fit_transform(scaler.fit_transform(red_train))
    new_red_val = pca.transform(scaler.transform(red_val))
    new_red_test = pca.transform(scaler.transform(red_test))
    joblib.dump(pca, os.path.join(model_path,(label+'_r_pca.pkl')))
    
    # Fit the pca to the scaled green training dataset, and transform
    # Transform the validation and test datasets using the fitted pca    
    new_green_train = pca.fit_transform(scaler.fit_transform(green_train))
    new_green_val = pca.transform(scaler.transform(green_val))
    new_green_test = pca.transform(scaler.transform(green_test))
    joblib.dump(pca, os.path.join(model_path,(label+'_g_pca.pkl')))
    
    # Fit the pca to the scaled blue training dataset, and transform
    # Transform the validation and test datasets using the fitted pca    
    new_blue_train = pca.fit_transform(scaler.fit_transform(blue_train))
    new_blue_val = pca.transform(scaler.transform(blue_val))
    new_blue_test = pca.transform(scaler.transform(blue_test))
    joblib.dump(pca, os.path.join(model_path,(label+'_b_pca.pkl')))
    
    # Concatenate the newly reduced datasets together for training,
    # validation, and test datasets
    print('Concatenating new data...')
    new_train = np.concatenate((new_red_train,new_green_train,new_blue_train),axis=1)
    new_val = np.concatenate((new_red_val,new_green_val,new_blue_val),axis=1)
    new_test = np.concatenate((new_red_test,new_green_test,new_blue_test),axis=1)

    # Change to the target folder
    os.chdir(pca_path)
    
    # Save the new arrays
    print('Saving data...')
    np.save(label+'_train_rgb',new_train)
    np.save(label+'_val_rgb',new_val)
    np.save(label+'_test_rgb',new_test)