# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:10:14 2018

@author: Tahlia
"""

import os
import pandas as pd
from data_config import data_config as dc
from PIL import Image
import numpy as np

data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
os.chdir(data_path)
df = pd.read_csv('attribute_list.csv',header = 1)

#Remove any files that are considered noisy i.e. the landscape photos
print('Removing noise...')
df_labels = dc.remove_noise(data_path,df)
train_labels = df_labels.iloc[0:3500,:]
test_labels = df_labels.iloc[3501:,:]

#Convert leftover images to a nested array where the shape is the number of images
#by the height*width of the images, by 3 (RGB)
#print('Convertng images to data...')
#image_data = dc.image_to_data_3_chan(data_path,df_labels)

print('Creating label and data slices...')
#train_data = dc.image_to_data_3_chan(data_path,train_labels)
train_data_gray = dc.image_to_data_gray(data_path,train_labels)
train_data_gray.tofile('trdg.dat')

#test_data = dc.image_to_data_3_chan(data_path,test_labels)
test_data_gray = dc.image_to_data_gray(data_path, test_labels)
test_data_gray.tofile('tedg.dat')
train_label = train_labels['smiling']
test_label = test_labels['smiling']

# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.utils.np_utils import to_categorical
# from keras import optimizers

# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

train_label_one_hot = to_categorical(train_label)
test_label_one_hot = to_categorical(test_label)
np.array(train_label_one_hot).tofile('trlsmile.dat')
np.array(test_label_one_hot).tofile('telsmile.dat')
