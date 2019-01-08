# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 04:15:07 2019

@author: Tahlia
"""

from run_model import run_model as rm

labels = ['smiling','eyeglasses','human','young','hair_color']

data_path ='D:/Tahlia/OneDrive/University/Year 4/Applied Machine Learning'
data_type = 'rgb'

fig=0
for label in labels:
    rm.plot_training(label,data_type,data_path,fig)
    fig=fig+1