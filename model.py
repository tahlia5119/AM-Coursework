# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:56:43 2018

@author: Tahlia
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.utils.np_utils import to_categorical
from keras import optimizers
from data_config import data_config as dc

data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
os.chdir(data_path)

print('Assigning labels...')
train_labels = pd.read_csv('train_labels.csv',header=0,index_col=0)
train_label = train_labels['smiling']
train_data_gray = dc.image_to_data_gray(data_path,train_labels)
test_labels = pd.read_csv('test_labels.csv',header=0,index_col=0)
test_label = test_labels['smiling']
test_data_gray = dc.image_to_data_gray(data_path, test_labels)

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

train_label_one_hot = to_categorical(train_label)
test_label_one_hot = to_categorical(test_label)

def createModelGray():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(256,256,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
     
    return model

print('Creating model...')
model = createModelGray()
batch_size = 10
epochs = 1

print('Compiling model...')
model.compile(optimizer = sgd, loss='binary_crossentropy',metrics=['accuracy'])

print('Fitting model...')
history = model.fit(train_data_gray, train_label_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, 
                    validation_data=(test_data_gray, test_label_one_hot))
model.evaluate(test_data_gray, test_label_one_hot)
