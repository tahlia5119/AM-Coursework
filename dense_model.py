# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:54:05 2019

@author: Tahlia
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:56:43 2018

@author: Tahlia
"""

import os
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.utils.np_utils import to_categorical
from keras import optimizers
from data_config import data_config as dc

label = 'young' #choose from 'smiling', 'eyeglasses', 'human', 'young', or 'hair_color'
data_type = 'rgb_pca' #choose from 'rgb', 'rgb_pca', or 'gray'
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
script_path = os.path.join(data_path)
os.chdir(data_path)

print('Getting data...')

if data_type != 'rgb_pca':
    train_data = np.load(os.path.join(data_path,'npy_files',label+'_train_'+data_type+'.npy'))
    test_data = np.load(os.path.join(data_path,'npy_files',label+'_test_'+data_type+'.npy'))
else:
    train_data = np.load(os.path.join(data_path,'npy_pca_files',label+'_train_rgb.npy')) 
    test_data = np.load(os.path.join(data_path,'npy_pca_files',label+'_test_rgb.npy'))

train_label = np.load(os.path.join(data_path,'npy_files',label+'_train_label.npy'))
test_label = np.load(os.path.join(data_path,'npy_files',label+'_test_label.npy'))
    
num_classes = train_label.shape[1]
input_shape = train_data.shape[1:]

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

def createModel():
    model = Sequential()
    model.add(Dense(200, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #model.add(Flatten())
    #model.add(Dense(50))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

print('Creating model...')
model = createModel()
batch_size = 500
epochs = 1

print('Compiling model...')
model.compile(optimizer = sgd, loss='categorical_crossentropy',metrics=['accuracy'])

start = time.time()
print('Fitting model...')
history = model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, verbose=1, 
                    validation_data=(test_data, test_label))
run = round(time.time()-start,2)
model.evaluate(test_data, test_label)
print('Fitting time: ',run)