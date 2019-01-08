# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:03:21 2019

@author: Tahlia
"""

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Main folder path
data_path = 'D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'

# Location of the scripts
script_path = os.path.join(data_path,'scripts')

# Location of the original npy files
npy_path = os.path.join(data_path, 'npy_files') 

# Location where to save new npy files after pca
pca_path = os.path.join(data_path, 'npy_pca_files')

labels = ['smiling','eyeglasses','human','young','hair_color']

# Reducing the pixel count feature set from 768 to 136 (similar to
# length of the facial landmarks feature set)
pca = PCA(n_components = 136)

#Initialize scaler
scaler = StandardScaler()

for label in labels:
    
    # Load the original npy files
    print('Loading files...')
    train = np.load(os.path.join(npy_path,label+'_train_pixel.npy'))
    val = np.load(os.path.join(npy_path,label+'_val_pixel.npy'))
    test = np.load(os.path.join(npy_path,label+'_test_pixel.npy'))
    
    # Scale the data, fit the pca to the training data and
    # transform the datasets with the fitted pca
    print('Scaling data and performing PCA...')
    new_train = pca.fit_transform(scaler.fit_transform(train))
    new_val = pca.transform(scaler.transform(val))
    new_test = pca.transform(scaler.transform(test))

    # Change to the folder where the pca npy files are to be saved
    os.chdir(pca_path)
    
    # Save the new npy files
    print('Saving data...')
    np.save(label+'_train_pixel',new_train)
    np.save(label+'_val_pixel',new_val)
    np.save(label+'_test_pixel',new_test)
    
    