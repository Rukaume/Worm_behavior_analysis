#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:20:21 2020

@author: miyazakishinichi
"""
import os
import sys
import pandas as pd
from tkinter import messagebox

def csv_file_read(filepath):
    file_dir, file_name = os.path.split(filepath)
    base, ext = os.path.splitext(file_name)
    if ext == '.csv':
        data = pd.read_csv(filepath, index_col = 0)
        return data
    else:
        return messagebox.showinfo('error',
                            'selected file is not csv file')
    
def image_list_extraction(directory_path):
    filelist = os.listdir(directory_path)
    filelist = [i for i in filelist if os.path.splitext(i)[1] == '.jpg' \
            or os.path.splitext(i)[1] == '.png']
        
        
