# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:38:22 2019

@author: Tahlia
"""

import os
import numpy as np
import pandas as pd
np.random.seed(0)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib 

label = 'eyeglasses' #choose from 'smiling', 'eyeglasses', 'human', 'young', or 'hair_color'
data_type = 'rgb' #choose from 'rgb' or 'gray'

#Main folder path
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'

#location of all the scripts
script_path = os.path.join(data_path,'scripts')

# load the numpy arrays from this folder
npy_path = os.path.join(data_path,'npy_files') 

# save the models and scalers to this folder
model_path = os.path.join(data_path,'models_scalers')

# Initialize the encoder
encoder = OneHotEncoder(sparse=False, categories = 'auto')

#Change to main directory
os.chdir(data_path)

#Load the numpy arrays that are the training and validation datasets and the
#respective label files
print('Getting data...')
train_data = np.load(os.path.join(npy_path,label+'_train_'+data_type+'.npy'))
test_data = np.load(os.path.join(npy_path,label+'_val_'+data_type+'.npy'))
train_data = np.concatenate((train_data,test_data))

# Open the files containing the label into a dataframe
train_label = pd.read_csv(os.path.join(data_path,'train_test_csv','eyeglasses_train.csv'),header = 0,index_col=0)['eyeglasses']
test_label = pd.read_csv(os.path.join(data_path,'train_test_csv','eyeglasses_val.csv'),header = 0,index_col=0)['eyeglasses']
train_label = pd.concat([train_label,test_label])

# Fit the encoder and save it
encoder.fit(np.array(train_label).reshape(-1,1))
joblib.dump(encoder, os.path.join(model_path,('eyeglasses_encoder.pkl')))
train_label = encoder.transform(np.array(train_label).reshape(-1,1))
test_label = encoder.transform(np.array(test_label).reshape(-1,1))
input_shape = train_data.shape[1:]

#Optimizers SGD selected with parameters: 
sgd = optimizers.SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True)

#Define the model layout
def createModel():
    
    model = Sequential()
    
    # First convolutional layer, similar to that of AlexNet
    # Since BatchNormaliation is included before the activation,
    # the Conv2D attribute 'use_bias' is set to False
    model.add(Conv2D(96, (11, 11), input_shape=input_shape,strides=(4,4),padding='valid',use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))    
    
    # Max pooling
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
    
    # Second convolutional layer with BatchNormalization
    model.add(Conv2D(96, (8,8),strides=(1,1), padding='valid',use_bias=False))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Max pooling
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
    
    #Flatten the model before passing to the fully connected layer
    model.add(Flatten())
    
    # Fully connected layer of 1024 units followed by BatchNormalization
    model.add(Dense(1024,use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Dropout included to lower risk of overfitting
    model.add(Dropout(0.25))
    
    # Fully connected layer with dropout
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # OUTPUT LAYER
    # Fully connected layer with as many neurons as there different classes
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    return model

print('Creating model...')
# Initialize the model, use batch_size=50 and 50 epochs
model = createModel()
batch_size = 50
epochs = 50

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=5),ModelCheckpoint(filepath=os.path.join(model_path,'best_model_eyeglasses.h5'), monitor='val_loss', save_best_only=True)]

print('Compiling model...')
# Compile the model - use 'categorical_crossentropy' for hair_color
model.compile(optimizer = sgd, loss='binary_crossentropy',metrics=['accuracy'])

# Fit the model
history = model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_data, test_label),callbacks=callbacks)

# serialize model to JSON
model_json = model.to_json()
with open(os.path.join(model_path,"eyeglasses_model.json"), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(os.path.join(model_path,"eyeglasses_model.h5"))