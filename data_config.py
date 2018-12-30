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
import lab2_landmarks as l2

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
            
        landmark_features, file_id = l2.extract_features_labels(img_path,df)
        df['faces_detected'] = faces
        
        df = df[df.faces_detected > 0]
        df = df[df.file_name.isin(file_id)]
        
        return df[df.columns.tolist()[:-1]], landmark_features
    
    def pixel_counts(path,df):
        
        image_array = []
        
        for i in df['file_name']:
            img = Image.open(os.path.join(path,'dataset',(str(i)+'.png')) ) 
            img_hist = img.histogram()   
            image_array.append(np.array(img_hist))
        
        return np.array(image_array)
    
    def image_to_data_gray(path,df):
        
        gray_array = []
        
        for i in df['file_name']:
            img_path = os.path.join(path,'dataset',(str(i)+'.png'))
            img = image.img_to_array(image.load_img(img_path,target_size=None,interpolation='bicubic')) 
            img = img.astype('uint8')
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grey = grey.astype('uint8')
            gray_array.append(np.array(grey))
            
        shape0 = np.array(gray_array).shape[0]
        shape1 = np.array(gray_array).shape[1]
        shape2 = np.array(gray_array).shape[2]
            
        return np.array(gray_array).reshape([shape0,shape1,shape2,1])
    
    def image_to_data_rgb(path,df):
        
        rgb_array = []
        
        for i in df['file_name']:
            img_path = os.path.join(path,'dataset',(str(i)+'.png'))
            img = image.img_to_array(image.load_img(img_path, target_size=None,interpolation='bicubic')) 
            img.astype('uint8')
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype('uint8')
            rgb_array.append(np.array(rgb))
        
        return np.array(rgb_array)
    
#    def facial_landmark_values(path,df):
#        
#        feature_array = []
#        detector = dlib.get_frontal_face_detector()
#        faces = []
#        
#        for i in df['file_name']:
#            img_path = os.path.join(path,'dataset',(str(i)+'.png'))
#            img = image.img_to_array(image.load_img(img_path, target_size=None,interpolation='bicubic')) 
#            img.astype('uint8')
#            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#            rgb = rgb.astype('uint8')
#            
#            rects = detector(rgb,1)
#            num_faces = len(rects)
#            
#            face_areas = np.zeros((1,num_faces))
#            face_shapes = np.zeros((136, num_faces),dtype =np.int64)
#            
#            
#            feature_array.append(np.array(rgb))
#            
#        
#        return feature_array

        