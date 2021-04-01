# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:23:08 2020

@author: Miyazaki
to do

inputで分岐を作って
if input == 1:
    background image 
    curvature analysis
else:
    select result csv
    
のようにする
"""


import os, cv2
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter 
from tkinter import messagebox
import os.path
import time 
import numpy as np
from tqdm import tqdm
import scipy.stats
import datetime
from skimage.morphology import medial_axis, skeletonize
import matplotlib.patches as patches
from skimage import measure
from skimage.measure import label, regionprops
from functions.functions import rgb_to_gray,sliding_window,get_image_filelist
from functions.functions import adjust, preprocessing, curvature_calc, get_ROI
from functions.functions import make_directories, make_background_and_subtracted_images
from functions.functions import curvature_analysis,image_subtraction_threshold
from functions.functions import threshold_GUI,image_subtraction_analysis, image_subtraction_analysis_data_save
from functions.functions import lethargus_analyzer


#get data list and ROI (csv)
def obtain_images_and_ROI():
    imagelist, image_file_path = get_image_filelist() 
    ROI_list = get_ROI()
    return imagelist, image_file_path, ROI_list

def subtraction():
    #obtain file paths
    imagelist, image_file_path, ROI_list = obtain_images_and_ROI()
    #obtain date
    date = datetime.date.today()
    #make directories 
    make_directories(date)
    #directory
    os.chdir(image_file_path)
    t1 = time.time() 
    print("making a background image")
    #make background image and subtracted images 
    make_background_and_subtracted_images(imagelist,date)
    os.chdir(image_file_path)
    #calculate body curvature and bodysize
    print("calculating body curvatures and bodysizes")
    bodysize_list = curvature_analysis(0, ROI_list,date)
    os.chdir(image_file_path)
    #image subtraction analysis
    print("image subtraction analysis")
    sample_std, sample_mean = image_subtraction_threshold(imagelist)
    
    #set threshold
    threshold_ans = input("please put threshold.\n \
                         if you want to set threshold by GUI, just ENTER.")
    if threshold_ans == "":
        saved_threshold_pixel, threshold = threshold_GUI(imagelist, 
                                                         sample_std, 
                                                         sample_mean)
    else:
        threshold = int(threshold_ans)
        saved_threshold_pixel = threshold * sample_std + sample_mean
    datalist = image_subtraction_analysis(imagelist, ROI_list,saved_threshold_pixel)
    t2 = time.time()
    elapsed_time = t2-t1
    print(f"経過時間：{elapsed_time}")
    os.chdir(image_file_path)
    image_subtraction_analysis_data_save(datalist, image_file_path, elapsed_time, date, threshold)
    os.chdir("../")
    return datalist, bodysize_list
    
def lethargus_analysis(datalist, bodysize_list):
    lethargus_analyzer(datalist, bodysize_list)

def main():
    subtraction_ans = input("if you already have subtraction data, please put y\
                           then, subtraction step is skipped")
    if str(subtraction_ans) == "y":
        bodysize_list = input('bodysizeをコンマ区切りで入力 : ').split(",")
        bodysize_list = [int(i) for i in bodysize_list]
        root = tkinter.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename()
        filedir = os.path.dirname(filepath)
        os.chdir(filedir)
        datalist = pd.read_csv(filepath)
        lethargus_analyzer(datalist, bodysize_list)
    else:
        datalist, bodysize_list = subtraction()
        lethargus_analyzer(datalist, bodysize_list)
    

if __name__ == '__main__':
    main()
    



