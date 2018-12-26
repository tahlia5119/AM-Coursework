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

train_labels.to_csv('train_labels.csv')
test_labels.to_csv('test_labels.csv')