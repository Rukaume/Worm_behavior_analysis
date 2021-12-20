# -*- coding: utf-8 -*-
"""

"""

import cv2
import datetime
import os
import os.path
import sys
import time
import tkinter
from tkinter import filedialog
from tkinter import messagebox

import numpy as np
import pandas as pd
from PIL import Image
from numpy import ndarray
from tqdm import tqdm

# parameters
threshold = 3


def get_ROI():
    root = tkinter.Tk()
    root.withdraw()
    # initial directory
    # dir = "C:/Users/miyas/desktop"
    messagebox.showinfo('select file', 'select ROI file')
    ROI_file_path = tkinter.filedialog.askopenfilename()
    if ROI_file_path == "":
        messagebox.showinfo('cancel', "stop before ROI setting")
        sys.exit()
    roi_data = csv_file_read(ROI_file_path)
    roi_data['left'] = roi_data['BX']
    roi_data['right'] = roi_data['BX'] + roi_data['Width']
    roi_data['low'] = roi_data['BY']
    roi_data['high'] = roi_data['BY'] + roi_data['Height']
    worm1_roi = roi_data.loc[1]['left':'high']
    worm2_roi = roi_data.loc[2]['left':'high']
    worm3_roi = roi_data.loc[3]['left':'high']
    ROI_list = list([worm1_roi, worm2_roi, worm3_roi])
    return ROI_list


def csv_file_read(filepath):
    file_dir, file_name = os.path.split(filepath)
    base, ext = os.path.splitext(file_name)
    if ext == '.csv':
        data = pd.read_csv(filepath, index_col=0)
        return data
    else:
        return messagebox.showinfo('error',
                                   'selected file is not csv file')


def get_image_filelist():
    root = tkinter.Tk()
    root.withdraw()
    # initial directory
    messagebox.showinfo('selectfiles', 'select analyzing image')
    image_file_path = tkinter.filedialog.askopenfilename()
    image_directory = os.path.dirname(image_file_path)
    if image_file_path == "":
        messagebox.showinfo('cancel', 'stop before image setting')
        sys.exit()
    os.chdir(image_directory)
    filelist = os.listdir('.')
    filelist = [i for i in filelist if os.path.splitext(i)[1] == '.jpg' \
                or os.path.splitext(i)[1] == '.png' \
                or os.path.splitext(i)[1] == ".tiff" \
                or os.path.splitext(i)[1] == ".tif"]
    return filelist


def data_select():
    ROI_list = get_ROI()
    file_list: list[str] = get_image_filelist()
    number_of_images: int = len(file_list) - 1
    if number_of_images < 100:
        messagebox.showinfo('cancel', 'there are not enough amount of images')
    else:
        pass
    return ROI_list, file_list, number_of_images


def threshold_set(file_list, number_of_images):
    sampling_cycle = (number_of_images // 100) - 1
    threshold_list = []
    for p in tqdm(range(100)):
        sample_num = p * sampling_cycle
        sample_temp = file_list[sample_num]
        next_to_sample_temp = file_list[sample_num + 1]
        sample_img = np.array(Image.open(sample_temp).convert("L")).astype("int8")
        next_to_sample_temp_img = np.array(Image.open(next_to_sample_temp) \
                                           .convert("L")).astype("int8")
        if p == 0:
            sample_subtract_img = (sample_img - next_to_sample_temp_img) / 255
        else:
            temp_subtract_img = (sample_img - next_to_sample_temp_img) / 255
            sample_subtract_img = np.vstack([sample_subtract_img, temp_subtract_img])

    blur = cv2.GaussianBlur(sample_subtract_img, (5, 5), 0)
    sample_mean = sample_subtract_img.mean()
    sample_std = sample_subtract_img.std()
    threshold_pixel = threshold * sample_std + sample_mean

    return threshold_pixel


def image_subtraction_analysis(file_list, ROI_list, threshold_pixel):
    t1 = time.time()
    data_array: ndarray = np.array([0, 0, 0])
    for n in tqdm(range(len(file_list) - 1)):
        img1: ndarray = np.array(Image.open(file_list[n]).convert("L")).astype("int8")
        img2: ndarray = np.array(Image.open(file_list[(int(n) + 1)]).convert("L")).astype("int8")
        subtracted_image: ndarray = (img2 - img1) / 255
        blured_image: ndarray = cv2.GaussianBlur(subtracted_image, (3, 3), 0)
        local_data: ndarray = np.array([0, 0, 0])
        for i, m in enumerate(ROI_list):
            left, right, low, high = int(m['left']), \
                                     int(m['right']), int(m['low']), int(m['high'])
            subtracted_image_crop: ndarray = blured_image[low:high, left:right]
            local_data[i]: int = np.count_nonzero(subtracted_image_crop > threshold_pixel)
        data_array: ndarray = np.vstack((data_array, local_data))
    t2 = time.time()
    elapsed_time = t2 - t1
    print(f"経過時間：{elapsed_time}")
    return data_array, elapsed_time


def data_save(data_list, elapsed_time):
    s = pd.DataFrame(data_list)
    s.columns = ['Area1', 'Area2', 'Area3']
    s_name = os.path.basename("./")

    date = datetime.date.today()
    time = datetime.datetime.now()
    os.chdir('../')
    os.makedirs('./analyzed_data_{}'.format(date), exist_ok=True)
    os.chdir('./analyzed_data_{}'.format(date))
    s.to_csv('./locomotor_activity_th={0}.csv'.format(threshold))
    path_w = './readme.txt'
    contents = '\nanalyzed_date: {0}\nelapsed time: {1}\nthreshold: {2}'.format(time,
                                                                                elapsed_time,
                                                                                threshold)


    with open(path_w, mode="a") as f:
        f.write(contents)


def main():
    ROI_list, file_list, number_of_images = data_select()
    threshold_pixel = threshold_set(file_list, number_of_images)
    data_array, elapsed_time = image_subtraction_analysis(file_list, ROI_list, threshold_pixel)
    data_save(data_array, elapsed_time)
    sys.exit()


if __name__ == '__main__':
    main()
