# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:55:23 2018

@author: Tahlia
"""

import pandas as pd
import numpy as np
import os
from PIL import Image
#from sklearn.model_selection import cross_val_score, StratifiedKFold
#import dlib
import cv2
#import tensorflow as tf
from keras.preprocessing import image
#import lab2_landmarks as l2

class data_config:
    
    def __init__(self):
        pass
    
    # Convert the images to be represented by their pixel counts
    # i.e. the number of pixels for each value 0-255 for each colour
    # band, therefore total length of each generated array is 768
    # NOT SPATIAL DATA
    def pixel_counts(path,df):
        
        image_array = []
        
        for i in df['file_name']:
            img = Image.open(os.path.join(path, (str(i)+'.png')) ) 
            img_hist = img.histogram()   
            image_array.append(np.array(img_hist))
        
        return np.array(image_array)

    # Convert the image to an array and then convert those values to those
    # on the grayscale spectrum
    def image_to_data_gray(path,df,target_size):
        
        gray_array = []
        
        for i in df['file_name']:
            img_path = os.path.join(path, (str(i)+'.png'))
            img = image.img_to_array(image.load_img(img_path,target_size=target_size,interpolation='bicubic')) 
            img = img.astype('uint8')
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grey = grey.astype('uint8')
            gray_array.append(np.array(grey))
            
        shape0 = np.array(gray_array).shape[0]
        shape1 = np.array(gray_array).shape[1]
        shape2 = np.array(gray_array).shape[2]
        
        # Need to reshape the array so that it has 4 dimensions and can
        # be used as input to a CNN model
        return np.array(gray_array).reshape([shape0,shape1,shape2,1])
    
    # Convert the image to an array and then convert those values from GBR 
    # (as a result of the image being a PNG) to RGB
    # Does not need to be reshaped as the resulting array is already
    # four dimensional
    def image_to_data_rgb(path,df,target_size):
        
        rgb_array = []
        
        for i in df['file_name']:
            img_path = os.path.join(path,(str(i)+'.png'))
            img = image.img_to_array(image.load_img(img_path, target_size=target_size,interpolation='bicubic')) 
            img.astype('uint8')
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype('uint8')
            rgb_array.append(np.array(rgb))
        
        return np.array(rgb_array)
        