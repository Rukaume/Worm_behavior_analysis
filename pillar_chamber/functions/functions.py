#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:14:30 2020

@author: miyazakishinichi
"""
import os, cv2
import sys
import pandas as pd
import matplotlib.pyplot as plt
import tkinter 
from tkinter import messagebox
import os.path
import numpy as np
from tqdm import tqdm
import datetime
from PIL import Image, ImageTk
from skimage import measure
from skimage.morphology import medial_axis, skeletonize
import seaborn as sns
from skimage.measure import label, regionprops
import itertools

def csv_file_read(filepath):
    file_dir, file_name = os.path.split(filepath)
    base, ext = os.path.splitext(file_name)
    if ext == '.csv':
        data = pd.read_csv(filepath, index_col = 0)
        return data
    else:
        return messagebox.showinfo('error',
                            'selected file is not csv file')

def rgb_to_gray(src):
    # obtain individual values
    b, g, r = src[:,:,0], src[:,:,1], src[:,:,2]
     # RGB to gray
    return np.array(0.2989 * r + 0.5870 * g + 0.1140 * b, dtype='float32')


def ROI_set(ROI_file_path):
    roi_data = csv_file_read(ROI_file_path)
    roi_data['left'] = roi_data['BX']
    roi_data['right'] = roi_data['BX'] + roi_data['Width']
    roi_data['low'] = roi_data['BY']
    roi_data['high'] = roi_data['BY'] + roi_data['Height']
    roi = []
    [roi.append(roi_data.loc[i+1]['left':'high']) for i in range(len(roi_data))]
    return roi


def rgb_to_gray(src):
    # obtain individual values
    b, g, r = src[:,:,0], src[:,:,1], src[:,:,2]
     # RGB to gray
    return np.array(0.2989 * r + 0.5870 * g + 0.1140 * b, dtype='float32')

def sliding_window(img, patch_size =[3, 3], istep =1, jstep=1, scale =1.0):
    Ni, Nj=[3,3]
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1]- Nj, jstep):
            patch = img[i:i + Ni, j:j+Nj]
            index = np.array([i,j])
            yield index, patch
            
def csv_file_read(filepath):
    file_dir, file_name = os.path.split(filepath)
    base, ext = os.path.splitext(file_name)
    if ext == '.csv':
        data = pd.read_csv(filepath, index_col = 0)
        return data
    else:
        return messagebox.showinfo('error',
                            'selected file is not csv file')    

def csv_file_read_without_header(filepath):
    file_dir, file_name = os.path.split(filepath)
    base, ext = os.path.splitext(file_name)
    if ext == '.csv':
        data = pd.read_csv(filepath, header = None)
    else:
        return messagebox.showinfo('error',
                            'selected file is not csv file')    
    return data



def make_directories(date):
    os.makedirs('../analyzed_data_{}'.format(date), exist_ok = True)
    os.chdir('../analyzed_data_{}'.format(date))
    #posture analysis directories
    os.makedirs("./posture_analysis", exist_ok = True)
    os.makedirs("./posture_analysis/background_image", exist_ok = True)
    os.makedirs("./posture_analysis/background_subtracted", exist_ok = True)
    os.makedirs("./posture_analysis/result_csv", exist_ok = True)
    #image_subtraction directories
    os.makedirs("./image_subtraction_analysis", exist_ok = True)
    
def get_image_filelist():
    root = tkinter.Tk()
    root.withdraw()
    #initial directory
    dir = "C:/Users/miyas/desktop"
    messagebox.showinfo('selectfiles', 'select analyzing image')
    image_file_path = tkinter.filedialog.askopenfilename(initialdir = dir)
    image_directory = os.path.dirname(image_file_path)
    if image_file_path == "":
        messagebox.showinfo('cancel', 'stop before image setting')
        sys.exit()
    os.chdir(image_directory)
    filelist = os.listdir('.')
    filelist = [i for i in filelist if os.path.splitext(i)[1] == '.jpg' \
                or os.path.splitext(i)[1] == '.png']
    return filelist, image_directory


def get_ROI():
    root = tkinter.Tk()
    root.withdraw()
    #initial directory
    #dir = "C:/Users/miyas/desktop"
    messagebox.showinfo('selectfile', 'select ROI file')
    ROI_file_path = tkinter.filedialog.askopenfilename()#initialdir = dir)
    if ROI_file_path == "":
        messagebox.showinfo('cancel', 'stop before ROI setting')
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


def make_background_and_subtracted_images(imagelist,date):
    filenum = len(imagelist)
    pickup_num = np.random.randint(0, filenum, 100)
    random_image_list = []
    [random_image_list.append(cv2.imread(imagelist[i])) for i in pickup_num]
    medianFrame = np.median(random_image_list, axis=0).astype(dtype=np.uint8)
    cv2.imwrite('../analyzed_data_{}/posture_analysis/background_image/background.jpg'.format(date),medianFrame)
    background = medianFrame
    [cv2.imwrite('../analyzed_data_{0}/posture_analysis/background_subtracted/{1}.jpg'.format(date,str(i).zfill(5)),
             cv2.absdiff(cv2.imread(imagelist[i]),background)) for i in tqdm(range(filenum))]

def make_background_and_subtracted_images_PIL(imagelist,date):
    filenum = len(imagelist)
    pickup_num = np.random.randint(0, filenum, 100)
    random_image_list = [np.array(Image.open(imagelist[i]).convert("L")).\
                         astype("int8") for i in pickup_num]
    medianFrame = np.median(random_image_list, axis=0).astype(dtype=np.uint8)
    cv2.imwrite('../analyzed_data_{}/posture_analysis/background_image/background.jpg'\
                .format(date),medianFrame)
    background = medianFrame
    [cv2.imwrite('../analyzed_data_{0}/posture_analysis/background_subtracted/{1}.jpg'\
                 .format(date,str(i).zfill(5)),cv2.absdiff(cv2.imread(imagelist[i])\
                ,background)) for i in tqdm(range(filenum))]

    
def curvature_analysis(image_create, ROI_list,date):
    os.chdir('../analyzed_data_{}/posture_analysis/background_subtracted'.format(date))
    bodysize_list = []
    sub_image_list = os.listdir("./")
    sub_images = [i for i in sub_image_list if os.path.splitext(i)[1] == '.jpg']
    results = []
    result_worm1, result_worm2, result_worm3 = [],[], []
    results.append(result_worm1)
    results.append(result_worm2)
    results.append(result_worm3)
    for l in tqdm(range(len(sub_images))):
        image = cv2.imread("./{}".format(sub_images[l]))
        for m in range(len(ROI_list)):
            left, right, low, high = int(ROI_list[m]['left']),int(ROI_list[m]['right']),int(ROI_list[m]['low']),int(ROI_list[m]['high'])
            subimage = image[low:high,left:right]
            rectangle_list, worm_image, worm_size = preprocessing(subimage)
            minarray = np.array([rectangle_list[0], rectangle_list[1]])
            maxarray = np.array([rectangle_list[2], rectangle_list[3]])
            try:
                indices, img_patches = zip(*sliding_window(worm_image))
                end_list = []
                for j in range(len(img_patches)):
                    if np.sum(img_patches[j]) == 2 and img_patches[j][1][1] == 1:
                        end_list.append(indices[j]+1)
                    else:
                        pass
                if len(end_list)==2:
                    point = end_list[0]
                    end_point = end_list[1]
                    worm_shape = []
                    while all(point != end_point):
                        worm_shape.append(point)
                        worm_image[point[0], point[1]] = 0
                        box = worm_image[point[0]-1:point[0]+2, point[1]-1:point[1]+2]
                        index_true = np.where(box == 1)
                        if np.sum(box) == 0:
                            break
                        else:
                            pointr = point[0] + int(index_true[0][0])-1
                            pointc = point[1] + int(index_true[1][0])-1
                            point = [pointr, pointc]
                    min_tile = np.tile(minarray, (len(worm_shape), 1))
                    worm_shape = np.array(worm_shape) + min_tile
                    worm_shape = np.array(worm_shape)[::5]
                    if len(worm_shape)>15:
                    #curvature caluculation
                        curvature = curvature_calc(worm_shape)
                        ## create image
    
                        center = int(len(worm_shape)/2)
                        results[m].append([l, np.sum(curvature),len(worm_shape), worm_shape[center][0],worm_shape[center][1], worm_size])
                    else:
                        pass
                else:
                    pass
            except ValueError:
                pass
    result_df1 = pd.DataFrame(results[0],columns=['TIme_w1', 'Curvature_w1', 'Segments_w1', 'Centerx_w1', 'Centery_w1', 'worm_size_w1'])
    result_df2 = pd.DataFrame(results[1],columns=['Time_w2', 'Curvature_w2', 'Segments_w2', 'Centerx_w2', 'Centery_w2', 'worm_size_w2'])
    result_df3 = pd.DataFrame(results[2],columns=['Time_w3', 'Curvature_w3', 'Segments_w3', 'Centerx_w3', 'Centery_w3', 'worm_size_w3'])
    bodysize_list.append(result_df1["worm_size_w1"].mean())
    bodysize_list.append(result_df2["worm_size_w2"].mean())
    bodysize_list.append(result_df3["worm_size_w3"].mean())
    result_df1.to_csv('../result_csv/worm1_result.csv')
    result_df2.to_csv('../result_csv/worm2_result.csv')
    result_df3.to_csv('../result_csv/worm3_result.csv')
    return bodysize_list




def adjust(img, alpha=1.0, beta=0.0):
    dst = alpha * img + beta
    return np.clip(dst, 0, 255).astype(np.uint8)


def preprocessing(image):
    image = adjust(image, alpha = 1)
    median = cv2.medianBlur(image,11)
    median = rgb_to_gray(median)
    median = np.round(median).astype('uint8')
    ret2,th2 = cv2.threshold(median,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    labeled_image = measure.label(th2, background=0)
    count = np.bincount(np.ravel(labeled_image))
    #np.argsort[::-1] -> descending order
    #background -> label[0] because the biggest region is background
    worm_label = np.argsort(count)[::-1][1]
    worm = (labeled_image==worm_label)
    skeleton = skeletonize(th2)
    skeleton_worm = np.logical_and(skeleton, worm)
    #search head or tail by sliding window
    minr, minc, maxr, maxc = regionprops(labeled_image)[worm_label-1].bbox
    worm_size = np.sum(th2[minr-1: maxr+1, minc-1:maxc+1])
    rectangle_list = [minr, minc, maxr, maxc]
    #extract worm padding 1 pixel
    worm_image = skeleton[minr-1: maxr+1, minc-1:maxc+1]
    return rectangle_list, worm_image, worm_size 
    
def curvature_calc(positions):
    dx_dt = np.gradient(positions[:, 0])
    dy_dt = np.gradient(positions[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    return curvature



def image_subtraction_threshold(image_list):
    image_number = len(image_list)-1

    if image_number < 100:
        messagebox.showinfo('cancel', 'there are not enough amount of images')
    else:
        pass
        
    sampling_cycle = (image_number // 100) -1
    #threshold_list = []
    for p in tqdm(range(100)):
        sample_num = p * sampling_cycle
        sample_temp = image_list[sample_num] 
        next_to_sample_temp = image_list[sample_num + 1]
        sample_img = np.array(Image.open(sample_temp).convert("L")).astype("int8")
        next_to_sample_temp_img = np.array(Image.open(next_to_sample_temp)\
                                           .convert("L")).astype("int8")
        if p == 0:
            sample_subtract_img = (sample_img - next_to_sample_temp_img)/255
        else:
            temp_subtract_img = (sample_img - next_to_sample_temp_img)/255
            sample_subtract_img = np.vstack([sample_subtract_img, temp_subtract_img])
    #blur = cv2.GaussianBlur(sample_subtract_img,(5,5),0)#cv2.medianBlur(sample_subtract_img,3)
    sample_mean = sample_subtract_img.mean()
    sample_std = sample_subtract_img.std()
    #threshold_pixel = float(1) * sample_std + sample_mean
    return sample_std, sample_mean

def threshold_GUI(filelist, sample_std, sample_mean):
    root2 = tkinter.Tk()
    root2.title(u"Threshold setting")
    root2.geometry("800x600")
    frame = tkinter.Frame(root2)
    frame.pack()
    
    image1 = np.array(Image.open(filelist[0]).convert("L")).astype("int8")
    image2 = np.array(Image.open(filelist[1]).convert("L")).astype("int8")
    subtract_image = (image1 - image2)/255
    blured = cv2.GaussianBlur(subtract_image,(3,3),0)
    image_pil = Image.fromarray(blured) 
    image_tk  = ImageTk.PhotoImage(image_pil, master = root2)
    
    canvas = tkinter.Canvas(frame, bg="black", width=600, height=450)
    canvas.grid(row = 0, column = 0)
    #canvas.pack()
    canvas.create_image(0, 0, image=image_tk, anchor=tkinter.NW, tag = "img")
    
    
    def get_threshold():
        canvas.delete("img")
        threshold = spinbox.get()
        new_threshold_pixel = int(threshold) * sample_std + sample_mean
        global new_image, new_image_pil, new_image_tk
        new_image = np.where(blured<new_threshold_pixel, 0, 255)
        new_image_pil = Image.fromarray(new_image)
        new_image_tk = ImageTk.PhotoImage(new_image_pil, master = root2)
        canvas.create_image(0, 0, image=new_image_tk,
                            anchor=tkinter.NW, tag = "img")
    
    def save_threshold():
        global saved_threshold_pixel, threshold
        threshold = spinbox.get()
        saved_threshold_pixel = int(threshold) * sample_std + sample_mean
        
    def close():
        root2.quit()
        root2.destroy()
    
    
    
    spinbox = tkinter.Spinbox(frame, from_ = 1, to=10, increment = 1,
                              state = "readonly")
    spinbox.grid(row = 1, column =0)
    #spinbox.pack()
    
    button1 = tkinter.Button(frame,text="apply",fg="black",width = 8,
                             command = get_threshold)
    button1.grid(row = 2, column = 0, padx=5, pady=5, sticky = "w")
    #button1.pack(anchor = "s",side = tkinter.BOTTOM)
    
    button2 = tkinter.Button(frame,text="set",fg="black",width = 8,
                             command = save_threshold)
    button2.grid(row = 2, column = 1, padx=5, pady=5, sticky = "w")
    #\button2.pack(anchor = "s",side = tkinter.BOTTOM)
    
    button3 = tkinter.Button(frame,text="execute",fg="black",width = 8,
                             command = close)
    button3.grid(row = 2, column = 2, padx=5, pady=5, sticky = "w")
    #button3.pack(anchor = "s",side = tkinter.BOTTOM)
    
    root2.mainloop()
    
    return saved_threshold_pixel, threshold

def image_subtraction_analysis(filelist, ROI_list,saved_threshold_pixel):
    datalist = [0,0,0]
    for n in tqdm(range(len(filelist)-1)):
        img1 = np.array(Image.open(filelist[n]).convert("L")).astype("int8")
        img2 = np.array(Image.open(filelist[(int(n)+1)]).convert("L")).astype("int8")
        subimage = (img2 - img1)/255
        blured = cv2.GaussianBlur(subimage,(3,3),0)
        localdata = []
        for m in ROI_list:
            subimage_crop = 0
            left, right, low, high = int(m['left']),\
                int(m['right']),int(m['low']),int(m['high'])
            subimage_crop = blured[low:high,left:right]
            count = np.count_nonzero(subimage_crop > saved_threshold_pixel) 
            localdata = np.append(localdata, count)
        datalist = np.vstack((datalist, localdata))
    return datalist

def image_subtraction_analysis_data_save(datalist, 
                                         image_directory, 
                                         elapsed_time, date,threshold):
    s = pd.DataFrame(datalist)
    s_name = os.path.basename(image_directory)
    s.columns = ['Area1', 'Area2',
                 'Area3']    
    time = datetime.datetime.now()
    os.chdir('../')
    os.chdir('./analyzed_data_{}/image_subtraction_analysis'.format(date))
    s.to_csv('./locomotor_activity_th={}.csv'.format(threshold))
    path_w = './readme.txt'
    contents = '\nanalyzed_date: {0}\nelapsed time: {1}\nthreshold:{2}'.format(time,
                                                                                elapsed_time, threshold)
    
    with open(path_w, mode = "a") as f:
        f.write(contents)
    #sys.exit()
        
def lethargus_analyzer(data_list, bodysize_list):
    data = pd.DataFrame(data_list)
    if len(data.columns) ==3 :
        pass
    elif len(data.columns) >= 4:
        data = data.loc[:,['Area1', 'Area2', 'Area3']]
    else: 
        print('The number of the columns is not adequate')
        sys.exit()
    
    
    try:
        data.columns = ['Area1', 'Area2', 'Area3']
    except ValueError as e:
        print(e)
        
    motiondata = data[["Area1", "Area2", "Area3"]]
    bodyratio = [i/100 for i in bodysize_list]

    Time = [a for a in range(len(motiondata))]
    #2秒に1度撮像しているので時間は行数の2倍になる。
    motiondata['Time'] = Time
    motiondata = motiondata.set_index(['Time'])
    #Timeをインデックスに
    motiondata['Area1_QorA'] = motiondata['Area1'] < bodyratio[0]
    motiondata['Area2_QorA'] = motiondata['Area2'] < bodyratio[1]
    motiondata['Area3_QorA'] = motiondata['Area3'] < bodyratio[2]

    motiondata.to_csv('./motiondata.csv')
    FoQ1, FoQ2, FoQ3 = [], [], []
    for j in range(len(motiondata)):
        tempdata1 = motiondata['Area1_QorA'][j:j+299]
        tempdata2 = motiondata['Area2_QorA'][j:j+299]
        tempdata3 = motiondata['Area3_QorA'][j:j+299]
        FoQ1.append(sum(tempdata1)/300)
        FoQ2.append(sum(tempdata2)/300)
        FoQ3.append(sum(tempdata3)/300)
    #motiondata['FoQ1'], motiondata['FoQ2'], motiondata['FoQ3'] = FoQ1, FoQ2, FoQ3
    FoQlist = []
    FoQlist.append(FoQ1)
    FoQlist.append(FoQ2)
    FoQlist.append(FoQ3)
    FoQnp = np.array(FoQlist)
    FoQdf = pd.DataFrame(FoQnp.T, columns= ['FoQ1', 'FoQ2', 'FoQ3'])
    #ここまででFoQのみのdataframe作成、保存
    FoQdf.to_csv('./FoQdf.csv')
    FoQvalue = pd.DataFrame(FoQnp.T, columns= ['FoQ1', 'FoQ2', 'FoQ3'])
    
    ####Quiescence Graph####
    sns.set()
    tempdf = FoQdf
    for k in tempdf.columns:
        plt.figure()
        tempdf[k].plot()
        plt.savefig('{}.png'.format(k))
        plt.show()
    
    FoQdf['FoQ1'] = FoQdf['FoQ1'] >= 0.05
    FoQdf['FoQ2'] = FoQdf['FoQ2'] >= 0.05
    FoQdf['FoQ3'] = FoQdf['FoQ3'] >= 0.05
    FoQdf.to_csv('./FoQdfbool.csv')


    Motion_boolean = pd.concat([motiondata['Area1_QorA'],
                            motiondata['Area2_QorA'],
                            motiondata['Area3_QorA']], axis = 1)
    During_lethargus_result = maxisland_start_len_mask(Motion_boolean)
    Q_durations = During_lethargus_result[3]
    Q_starts = During_lethargus_result[2]
    
    
    island_result = maxisland_start_len_mask(FoQdf)
    
    #resultは各カラムの最大islandのスタート位置のリスト、長さのリスト、すべてのislandのスタート位置のリスト
    #Activeresult is for active bout analysis
    
    #すべてのislandの長さのリストのtupleとなっている。
    max_start, max_length, all_start, all_length = island_result[0], island_result[1], island_result[2], island_result[3] 
    print(all_start)
    ###データを各列に分ける###
    columnnum = len(FoQdf)
    
    
    ####For Active bout####
    motiondata['Area1_QorA'] = motiondata['Area1'] >= bodyratio[0]
    motiondata['Area2_QorA'] = motiondata['Area2'] >= bodyratio[1]
    motiondata['Area3_QorA'] = motiondata['Area3'] >= bodyratio[2]
    
    motiondata.to_csv('./motiondata_for_active.csv')
    Motion_boolean_active = pd.concat([motiondata['Area1_QorA'],
                                motiondata['Area2_QorA'],
                                motiondata['Area3_QorA']], axis = 1)
    During_lethargus_result_active = maxisland_start_len_mask(Motion_boolean_active)
    A_durations = During_lethargus_result_active[3]
    A_starts = During_lethargus_result_active[2]
    
    
    #####About Area1#####
    #FoQ1の結果のインデックスの抜き出し
    FoQ1_index = list(itertools.chain.from_iterable(np.where(all_start <= columnnum)))
    #ここでFoQ1_indexはリストになっている。
    FoQ1_start= all_start[FoQ1_index]
    FoQ1_length = all_length[FoQ1_index]
    FoQ1_Lethargusnum = np.count_nonzero(FoQ1_length > 1800)
    #ファンシーインデックスによる指定
    max_end1 = max_start[0] + max_length[0]
    max_end1_out = max_end1 + 300
    FoQ1average = sum(FoQ1[max_start[0]:max_end1])/max_length[0]
    FoQ1_out = sum(FoQ1[max_end1:max_end1_out])/300
    Lethargus1_length = max_length[0]/1800
    #チェック事項
    if max_start[0] < 1800:
        FoQ1judge = 'Lethargus開始が撮影開始1時間より前です'
        FoQ1num = 1
    elif FoQ1_Lethargusnum > 1:
        FoQ1judge = 'Lethargusが複数定義できます'
        FoQ1num = 1
    elif max_end1 > columnnum - 1800:
        FoQ1judge = 'Lethargusが終了するのが測定終了1時間前より後です。'
        FoQ1num = 1
    else:
        FoQ1judge = 'Lethargus解析に使用できます\n平均値は{0}でoutでの平均値は{1}'.format(FoQ1average, FoQ1_out)
        FoQ1num = 0
        LeFoQdf1 = FoQvalue["FoQ1"][max_start[0]-1800:]
        LeFoQdf1.to_csv('./LeFoQdf1.csv')
    
    #lethargus中のQ or Aの判定結果
    if FoQ1num == 0:
        Area1_Q_starts_index = np.where((Q_starts > max_start[0]) & (Q_starts < max_end1))
        Area1_A_starts_index = np.where((A_starts > max_start[0]) & (A_starts < max_end1))
        Area1_A_starts_lethargus = A_starts[Area1_A_starts_index]
        Area1_Q_starts_lethargus = Q_starts[Area1_Q_starts_index]
        Area1_Q_durations = Q_durations[Area1_Q_starts_index]
        Total_Q1 = np.sum(Area1_Q_durations)
        Area1_A_durations = A_durations[Area1_A_starts_index]
        Total_A1 = np.sum(Area1_A_durations)
        Mean_Area1_Q_duration = np.mean(Area1_Q_durations) * 2
        Mean_Area1_A_duration = np.mean(Area1_A_durations) * 2
        Area1_transitions = len(Area1_Q_durations) / Lethargus1_length
        # order A -> Q 
        if Area1_A_starts_lethargus[0] > Area1_Q_starts_lethargus[0]:
            Area1_Q_durations = Area1_Q_durations[1:]
        #if A is more than Q, A is deleted
        if len(Area1_A_durations) > len(Area1_Q_durations):
            Area1_A_durations = Area1_A_durations[:-1]
        Lethargus_QandA = np.stack((Area1_A_durations, Area1_Q_durations),1)
        Lethargus_QandA = pd.DataFrame(Lethargus_QandA, columns = ["A", "Q"])
        first_range = np.where(Area1_A_starts_lethargus<Area1_A_starts_lethargus[0] + 1800, True, False).sum()
        #Lethargus_QandA_first = Q and A durations for 1 hour from lethargus start
        Lethargus_QandA_first = Lethargus_QandA[0:first_range]
        Lethargus_QandA_second = Lethargus_QandA[first_range:]
        Lethargus_QandA_first.to_csv("Lethargus_QandA_worm1_first.csv")
        Lethargus_QandA_second.to_csv("Lethargus_QandA_worm1_second.csv")
    else:
        Mean_Area1_A_duration, Mean_Area1_Q_duration, Area1_transitions, Total_Q1,Total_A1 = 0,0,0, 0,0
    
    #FoQ2の結果
    FoQ2_index = np.where((columnnum < all_start) &(all_start <= columnnum*2))
    FoQ2_start, FoQ2_length = all_start[FoQ2_index],all_length[FoQ2_index]
    FoQ2_Lethargusnum = np.count_nonzero(FoQ2_length > 1800)
    max_end2 = max_start[1] + max_length[1]
    max_end2_out = max_end2 + 300
    FoQ2average = sum(FoQ2[max_start[1]:max_end2])/max_length[1]
    FoQ2_out = sum(FoQ2[max_end2:max_end2_out])/300
    Lethargus2_length = max_length[1]/1800
    if max_start[1] < 1800:
        FoQ2judge = 'Lethargus開始が撮影開始1時間より前です'
        FoQ2num = 1
    elif FoQ2_Lethargusnum > 1:
        FoQ2judge = 'Lethargusが複数定義できます'
        FoQ2num = 1
    elif max_end2 > columnnum*2 - 1800:
        FoQ2judge = 'Lethargusが終了するのが測定終了1時間前より後です。'
        FoQ2num = 1
    else:
        FoQ2judge = 'Lethargus解析に使用できます\n平均値は{0}でoutでの平均値は{1}'.format(FoQ2average, FoQ2_out)
        FoQ2num = 0
        LeFoQdf2 = FoQvalue["FoQ2"][max_start[0]-1800:]
        LeFoQdf2.to_csv('./LeFoQdf2.csv')
    
    if FoQ2num == 0:
        Area2_Q_starts_index = np.where((columnnum + max_start[1] < Q_starts)&(Q_starts <= columnnum + max_end2))
        Area2_A_starts_index = np.where((columnnum + max_start[1] < A_starts)&(A_starts <= columnnum + max_end2))
        Area2_A_starts_lethargus = A_starts[Area2_A_starts_index]
        Area2_Q_starts_lethargus = Q_starts[Area2_Q_starts_index]
        Area2_Q_durations = Q_durations[Area2_Q_starts_index]
        Area2_A_durations = A_durations[Area2_A_starts_index]
        Total_Q2 = np.sum(Area2_Q_durations)
        Area2_A_durations = A_durations[Area2_A_starts_index]
        Total_A2 = np.sum(Area2_A_durations)
        Mean_Area2_Q_duration = np.mean(Area2_Q_durations) * 2
        Mean_Area2_A_duration = np.mean(Area2_A_durations) * 2
        Area2_transitions = len(Area2_Q_durations) / Lethargus2_length
        # order A -> Q 
        if Area2_A_starts_lethargus[0] > Area2_Q_starts_lethargus[0]:
            Area2_Q_durations = Area2_Q_durations[1:]
        #if A is more than Q, A is deleted
        if len(Area2_A_durations) > len(Area2_Q_durations):
            Area2_A_durations = Area2_A_durations[:-1]
        Lethargus_QandA = np.stack((Area2_A_durations, Area2_Q_durations),1)
        Lethargus_QandA = pd.DataFrame(Lethargus_QandA, columns = ["A", "Q"])
        first_range = np.where(Area2_A_starts_lethargus<Area2_A_starts_lethargus[0] + 1800, True, False).sum()
        #Lethargus_QandA_first = Q and A durations for 1 hour from lethargus start
        Lethargus_QandA_first = Lethargus_QandA[0:first_range]
        Lethargus_QandA_second = Lethargus_QandA[first_range:]
        Lethargus_QandA_first.to_csv("Lethargus_QandA_worm2_first.csv")
        Lethargus_QandA_second.to_csv("Lethargus_QandA_worm2_second.csv")
    else:
        Mean_Area2_A_duration, Mean_Area2_Q_duration, Area2_transitions, Total_Q2, Total_A2 = 0,0,0, 0, 0
    
    #FoQ3の結果
    FoQ3_index = np.where(all_start > columnnum*2)
    FoQ3_start, FoQ3_length = all_start[FoQ3_index],all_length[FoQ3_index]
    FoQ3_Lethargusnum = np.count_nonzero(FoQ3_length > 1800)
    max_end3 = max_start[2] + max_length[2]
    max_end3_out = max_end3 + 300
    FoQ3average = sum(FoQ3[max_start[2]:max_end3])/max_length[2]
    FoQ3_out = sum(FoQ3[max_end3:max_end3_out])/300
    Lethargus3_length = max_length[2]/1800
    if max_start[2] < 1800:
        FoQ3judge = 'Lethargus開始が撮影開始1時間より前です'
        FoQ3num = 1
    elif FoQ3_Lethargusnum > 1:
        FoQ3judge = 'Lethargusが複数定義できます'
        FoQ3num = 1
    elif max_end3 > columnnum*3 - 1800:
        FoQ3judge = 'Lethargusが終了するのが測定終了1時間前より後です。'
        FoQ3num = 1
    else:
        FoQ3judge = 'Lethargus解析に使用できます\n平均値は{0}でoutでの平均値は{1}'.format(FoQ3average, FoQ3_out)
        FoQ3num = 0
        LeFoQdf3 = FoQvalue["FoQ3"][max_start[0]-1800:]
        LeFoQdf3.to_csv('./LeFoQdf3.csv')
        
    if FoQ3num == 0:
        Area3_Q_starts_index = np.where((columnnum*2 + max_start[2] < Q_starts)&(Q_starts <= columnnum*2 + max_end3))
        Area3_A_starts_index = np.where((columnnum*2 + max_start[2] < A_starts)&(A_starts <= columnnum*2 + max_end3))
        Area3_Q_durations = Q_durations[Area3_Q_starts_index]
        Area3_A_durations = A_durations[Area3_A_starts_index]
        Area3_A_starts_lethargus = A_starts[Area3_A_starts_index]
        Area3_Q_starts_lethargus = Q_starts[Area3_Q_starts_index]
        Total_Q3 = np.sum(Area3_Q_durations)
        Area3_A_durations = A_durations[Area3_A_starts_index]
        Total_A3 = np.sum(Area3_A_durations)
        Mean_Area3_Q_duration = np.mean(Area3_Q_durations) * 2
        Mean_Area3_A_duration = np.mean(Area3_A_durations) * 2
        Area3_transitions = len(Area3_Q_durations) / Lethargus3_length
        # order A -> Q 
        if Area3_A_starts_lethargus[0] > Area3_Q_starts_lethargus[0]:
            Area3_Q_durations = Area3_Q_durations[1:]
        #if A is more than Q, A is deleted
        if len(Area3_A_durations) > len(Area3_Q_durations):
            Area3_A_durations = Area3_A_durations[:-1]
        Lethargus_QandA = np.stack((Area3_A_durations, Area3_Q_durations),1)
        Lethargus_QandA = pd.DataFrame(Lethargus_QandA, columns = ["A", "Q"])
        first_range = np.where(Area3_A_starts_lethargus<Area3_A_starts_lethargus[0] + 1800, True, False).sum()
        #Lethargus_QandA_first = Q and A durations for 1 hour from lethargus start
        Lethargus_QandA_first = Lethargus_QandA[0:first_range]
        Lethargus_QandA_second = Lethargus_QandA[first_range:]
        Lethargus_QandA_first.to_csv("Lethargus_QandA_worm3_first.csv")
        Lethargus_QandA_second.to_csv("Lethargus_QandA_worm3_second.csv")
    else:
        Mean_Area3_A_duration, Mean_Area3_Q_duration, Area3_transitions,Total_Q3, Total_A3 = 0,0,0,0,0
    
    #ans = messagebox.askokcancel('lethargus_detector.py',
                                 #'ROI1: {0}\nROI2: {1}\nROI3: {2}'.format(FoQ1judge,
                                                                         # FoQ2judge, FoQ3judge))
    
    datas = ['bodysize', 'FoQ_during_Lethargus', 'FoQ_out' ,'duration',
             'interpletation 0 is adequate', 'Mean Quiescent Bout',
             'Mean Active Bout','Transitions', "Total Q", "Total A"]
    result1 = [bodysize_list[0], FoQ1average, FoQ1_out, Lethargus1_length, FoQ1judge,
               Mean_Area1_Q_duration,Mean_Area1_A_duration, Area1_transitions, Total_Q1, Total_A1]
    result2 = [bodysize_list[1], FoQ2average, FoQ2_out, Lethargus2_length, FoQ2judge,
               Mean_Area2_Q_duration,Mean_Area2_A_duration, Area2_transitions, Total_Q2, Total_A2]
    result3 = [bodysize_list[2], FoQ3average, FoQ3_out, Lethargus3_length, FoQ3judge,
               Mean_Area3_Q_duration,Mean_Area3_A_duration, Area3_transitions, Total_Q3, Total_A3]
    result = pd.DataFrame([result1, result2, result3],
                          index = ['worm1', 'worm2', 'worm3'], columns = datas)
    
    result.to_csv('./result_summary.csv')
## to do ##
    



def bout_detector(array):
    array = np.array(array)
    temp = np.arange(len(array))
    temp[array>0] = 0
    np.maximum.accumulate(temp, out = temp)
    left = temp
    array_r = array[::-1]
    temp = np.arange(len(array))
    temp[array_r>0] = 0
    np.maximum.accumulate(temp, out = temp)
    right = len(array)-1-temp[::-1]
    y = right 
    y-= left +1
    y[array==0] = 0
    return y

def maxisland_start_len_mask(a, fillna_index = -1, fillna_len = 0):
    # a is a boolean array

    pad = np.zeros(a.shape[1],dtype=bool)
    mask = np.vstack((pad, a, pad))

    mask_step = mask[1:] != mask[:-1]
    idx = np.flatnonzero(mask_step.T)
    island_starts = idx[::2]
    island_lens = idx[1::2] - idx[::2]
    n_islands_percol = mask_step.sum(0)//2

    bins = np.repeat(np.arange(a.shape[1]),n_islands_percol)
    scale = island_lens.max()+1

    scaled_idx = np.argsort(scale*bins + island_lens)
    grp_shift_idx = np.r_[0,n_islands_percol.cumsum()]
    max_island_starts = island_starts[scaled_idx[grp_shift_idx[1:]-1]]

    max_island_percol_start = max_island_starts%(a.shape[0]+1)

    valid = n_islands_percol!=0
    cut_idx = grp_shift_idx[:-1][valid]
    max_island_percol_len = np.maximum.reduceat(island_lens, cut_idx)

    out_len = np.full(a.shape[1], fillna_len, dtype=int)
    out_len[valid] = max_island_percol_len
    out_index = np.where(valid,max_island_percol_start,fillna_index)
    return out_index, out_len, island_starts, island_lens

