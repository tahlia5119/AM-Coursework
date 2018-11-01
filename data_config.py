# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:55:23 2018

@author: Tahlia
"""

import pandas as pd
import os
from PIL import Image

class data_config:
    
    def __init__(self):
        pass
    
    def remove_noise(df):
        #The noise removal is based on the attribute 'hair_color' as this is the
        #only multiclass attribute - any files with a hair_color label of '-1' does
        #not correspond to any hair colour, including bald. Therefore, these files
        #need to be removed from the attribute list
        df = df[df.hair_color != -1]
        return df
    
    def image_to_data(path,df):
        
        dict_images = {n: [] for n in df['file_name']}

        for i in df['file_name']:
            img = Image.open(os.path.join(path,'dataset',(str(i)+'.png')) ) 
            img_hist = img.histogram()     
            dict_images[i] = img_hist
        
        return pd.DataFrame.from_dict(dict_images,orient='index')
    
#    def augment_images_bin(image_hist,label):
#        df = pd.concat(image_hist,label,axis=1,ignore_index=True)
#        no = label.value_counts()[-1]
#        yes = label.value_counts()[1]
#        perc = abs(no-yes)/no
#        
#        if perc < 0.5 or perc > 1.0:
#            change = min(no,yes)
#            df_change = df[df[label.name]==change]
            
            
        