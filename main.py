# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:10:14 2018

@author: Tahlia
"""

import os
import pandas as pd
from data_config import data_config as dc
from PIL import Image

data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
os.chdir(data_path)
df = pd.read_csv('attribute_list.csv',header = 1)

#Remove any files that are considered noisy i.e. the landscape photos
df_labels = dc.remove_noise(df)

#Convert leftover images to an array of values and store in a dictionary, which
#is then converted to a dataframe

df_image_data = dc.image_to_data(data_path,df_labels)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

#skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
#svc = SVC(random_state = 0)
#mlp = MLPClassifier(random_state = 0)
#lr = LogisticRegression(random_state=0)
#
#label = df_labels['smiling']
#print(cross_val_score(lr, df_image_data, label, cv=skf,verbose=10))

#Test different sklearn classifiers to narrow down the options by comparing their CV scores
classifiers = [MLPClassifier(random_state=0),
               SVC(random_state=0),
               LogisticRegression(random_state=0),
               RandomForestClassifier(random_state=0),
               AdaBoostClassifier(random_state=0),
               DecisionTreeClassifier(random_state=0)]

names = ['MLP','SVC','LR','RF','Ada','DT']

dict_dict, cv_dict = dc.tabulate_cvs(classifiers,names,df_labels,df_image_data)

cv_df = pd.DataFrame.from_dict(cv_dict, orient = 'index', columns = list(df_labels.columns.get_values())[1:])
