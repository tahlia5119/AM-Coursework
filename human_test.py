# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 23:05:56 2019

@author: Tahlia
"""


import os
import numpy as np
import pandas as pd

from sklearn.externals import joblib 

# Use balanced_accuracy_score to look at performance since there will be an imbalance of classes
from sklearn.metrics import balanced_accuracy_score

from keras.models import model_from_json

from keras import optimizers

label = 'human'
data_type = 'rgb'

#Main folder path
data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'

#location of all the scripts
script_path = os.path.join(data_path,'scripts')

# load the numpy arrays from this folder
npy_path = os.path.join(data_path,'npy_files') 

# Load encoder
encoder = joblib.load(os.path.join(data_path,'models_scalers','human_encoder.pkl'))

# Load sgd optimizer
sgd = optimizers.SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True)

# load json and create model
json_file = open(os.path.join(data_path,'models_scalers','human_model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join(data_path,'models_scalers','best_model_human.h5'))
print("Loaded model from disk")
 
print('Getting data...')
x_test = np.load(os.path.join(npy_path,label+'_test_'+data_type+'.npy'))

# Open the files containing the label into a dataframe
y_test = pd.read_csv(os.path.join(data_path,'train_test_csv',label+'_test.csv'),header = 0,index_col=0)[label]
y_test_oh = encoder.transform(np.array(y_test).reshape(-1,1))

# Make predictions on the test data
predictions = loaded_model.predict(x_test)

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Score the test data
score = loaded_model.evaluate(x_test,y_test_oh)[1]

# Get balanced accuracy score
balanced = balanced_accuracy_score(y_test,encoder.inverse_transform(predictions))


