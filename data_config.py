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
import dlib
import cv2
#import tensorflow as tf
from keras.preprocessing import image
#import lab2_landmarks as l2

class data_config:
    
    def __init__(self):
        pass
    
    def remove_noise(path,df):
        #The noise removal uses the dlib and cv2 libraries to detect faces and remove any
        #images where the length of the returned array is 0
        #A dataframe is returned that consists of the file_name, hair_color,
        #eyeglasses, smiling, young, and human labels for the images that 
        detector = dlib.get_frontal_face_detector()
        faces = []
        
        for i in df['file_name']:
            img_path = os.path.join(path,'dataset',(str(i)+'.png'))
            img = image.img_to_array(image.load_img(img_path,target_size=None,interpolation='bicubic')) 
            img = img.astype('uint8')
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype('uint8')
            dets=detector(rgb,1)
            faces.append(len(dets))
            
        df['faces_detected'] = faces
        
        df = df[df.faces_detected > 0]
        
        return df[df.columns.tolist()[:-1]]
    
    def pixel_counts(path,df):
        
        image_array = []
        
        for i in df['file_name']:
            img = Image.open(os.path.join(path,'dataset',(str(i)+'.png')) ) 
            img_hist = img.histogram()   
            image_array.append(np.array(img_hist))
        
        return np.array(image_array)
    
    def image_to_data_gray(path,df):
        
        chan_array = []
        
        for i in df['file_name']:
            img_path = os.path.join(path,'dataset',(str(i)+'.png'))
            img = image.img_to_array(image.load_img(img_path,target_size=None,interpolation='bicubic')) 
            img = img.astype('uint8')
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grey = grey.astype('uint8')
            chan_array.append(np.array(grey))
            
            shape0 = np.array(chan_array).shape[0]
            shape1 = np.array(chan_array).shape[1]
            shape2 = np.array(chan_array).shape[2]
            
        return np.array(chan_array).reshape([shape0,shape1,shape2,1])
    
    def image_to_data_3_chan(path,df):
        
        chan_3_array = []
        
        for i in df['file_name']:
            img = image.load_img(os.path.join(path,'dataset',(str(i)+'.png'))) 
            chan_3_array.append(np.array(img))
        
        #Reshape the array
        shape0 = np.array(chan_3_array).shape[0]
        shape1 = np.array(chan_3_array).shape[1]*np.array(chan_3_array).shape[2]
        shape2 = np.array(chan_3_array).shape[3]
        
        return np.array(chan_3_array)#np.array(chan_3_array).reshape([shape0, shape1, shape2])
    
    def facial_landmark_values(path,df):
        
                
        
        