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

data_path = 'D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
script_path = os.path.join(data_path,'scripts')
npy_path = os.path.join(data_path, 'npy_files') 
pca_path = os.path.join(data_path, 'npy_pca_files')

labels = ['smiling','eyeglasses','human','young','hair_color']

pca = PCA(n_components = 50)
scaler = StandardScaler()

for label in labels:
    
    print('Loading files and reshaping...')
    train = np.load(os.path.join(npy_path,label+'_train_rgb.npy'))
    test = np.load(os.path.join(npy_path,label+'_test_rgb.npy'))
    
    train_reshape = train.reshape(train.shape[0],np.prod(train.shape[1:-1]),3)
    test_reshape = test.reshape(test.shape[0],np.prod(test.shape[1:-1]),3)
    red_train = train_reshape[:,:,0]
    green_train = train_reshape[:,:,1]
    blue_train = train_reshape[:,:,2]
    red_test = test_reshape[:,:,0]
    green_test = test_reshape[:,:,1]
    blue_test = test_reshape[:,:,2]
    
    print('Scaling data and performing PCA...')
    new_red_train = pca.fit_transform(scaler.fit_transform(red_train))
    new_red_test = pca.transform(scaler.transform(red_test))
    new_green_train = pca.fit_transform(scaler.fit_transform(green_train))
    new_green_test = pca.transform(scaler.transform(green_test))
    new_blue_train = pca.fit_transform(scaler.fit_transform(blue_train))
    new_blue_test = pca.transform(scaler.transform(blue_test))
    
    print('Concatenating new data...')
    new_train = np.concatenate((new_red_train,new_green_train,new_blue_train),axis=1)
    new_test = np.concatenate((new_red_test,new_green_test,new_blue_test),axis=1)

    os.chdir(pca_path)
    
    print('Saving data...')
    np.save(label+'_train_rgb',new_train)
    np.save(label+'_test_rgb',new_test)
    
    
    
    
    
    