# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:56:43 2018

@author: Tahlia
"""

import os
import numpy as np
import time
np.random.seed(0)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from data_config import data_config as dc
from keras.metrics import categorical_accuracy
from matplotlib import pyplot
from keras.callbacks import EarlyStopping, ModelCheckpoint

label = 'young' #choose from 'smiling', 'eyeglasses', 'human', 'young', or 'hair_color'
data_type = 'rgb' #choose from 'rgb' or 'gray'

#Main folder path
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'

#location of all the scripts
script_path = os.path.join(data_path,'scripts')

#save the numpy arrays to this folder
npy_path = os.path.join(data_path,'npy_files') 

#File name and loction for the learning curve that is output
graph_path = os.path.join(data_path,'train_curves','sgd6'+label+'_'+data_type+'.png')

#Change to main directory
os.chdir(data_path)

#Load the numpy arrays that are the training and validation datasets and the
#respective label files
print('Getting data...')
train_data = np.load(os.path.join(npy_path,label+'_train_'+data_type+'.npy'))
test_data = np.load(os.path.join(npy_path,label+'_val_'+data_type+'.npy'))
train_label = np.load(os.path.join(data_path,'npy_files',label+'_train_label.npy'))
test_label = np.load(os.path.join(data_path,'npy_files',label+'_val_label.npy'))

#Get the number of classes (for the final Dense layer)
num_classes = train_label.shape[1]
input_shape = train_data.shape[1:]

#Optimizers (SGD selected with parameters: 
#lr=.0001, decay=1e-6, momentum=0.9, nesterov=True
sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.adam(lr=0.001, beta_1=0.9,beta_2=0.999,epsilon=1e-8)

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
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    return model

print('Creating model...')
# Initialize the model, use batch_size=50 and 50 epochs
model = createModel()
batch_size = 50
epochs = 50

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss'),ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

print('Compiling model...')
# Compile the model - use 'categorical_crossentropy' for hair_color
if label == 'hair_color':
    model.compile(optimizer = sgd, loss='categorical_crossentropy',metrics=['accuracy'])
else:
    model.compile(optimizer = sgd, loss='binary_crossentropy',metrics=['accuracy'])

# Meausre fitting time
start = time.time()
print('Fitting model...')

# Fit the model
history = model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, verbose=1, 
                    validation_data=(test_data, test_label))#,callbacks=callbacks)

# Fitting time
run = round(time.time()-start,2)

# Evaluate the model using the validation set
#model.evaluate(test_data, test_label)
print('Fitting time: ',run)

# plot train and validation loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
#pyplot.show()
pyplot.savefig(graph_path)