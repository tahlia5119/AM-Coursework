# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 09:52:42 2018

@author: Tahlia
"""

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling2D, Dropout, Activation, Conv2D
from keras.layers.embeddings import Embedding
import os
import pandas as pd
from data_config import data_config as dc
from PIL import Image
import numpy as np

data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
os.chdir(data_path)
df = pd.read_csv('attribute_list.csv',header = 1)

#Remove any files that are considered noisy i.e. the landscape photos
df_labels = dc.remove_noise(df)

label_array = np.array(df_labels['smiling'].values)

#Create a numpy array of 3 channel images resized to (227,227,3)
image_data = dc.image_to_data_3_chan(data_path,df_labels)

#Start a sequential model
#model = Sequential()
#
#model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(227,227,3)))
#model.add(BatchNormalization())
#model.add(Conv2D(32, kernel_size=3, activation='relu'))
#model.add(Flatten())
#model.add(Dense(1, activation='softmax'))
#
#model.summary()
#
## (4) Compile 
#model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
#
## (5) Train
#model.fit(image_data, label_array, batch_size=64, epochs=1, verbose=1, validation_split=0.2, shuffle=True)

hist_image_data = dc.image_to_data(data_path,df_labels)
hist_image = np.reshape(hist_image_data.values,(hist_image_data.shape[0],1,hist_image_data.shape[1],))
model = Sequential()
model.add(Flatten(input_shape=(3902,768)))
model.add(Dense(1,activation='relu'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(hist_image,label_array,batch_size=64,epochs=1,verbose=1, validation_split=0.2,shuffle=True)
