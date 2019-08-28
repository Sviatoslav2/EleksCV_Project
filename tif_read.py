import os
import numpy as np
import pandas as pd
from PIL import Image
import tifffile as tiff



def get_dirs(path):
    dct_of_files = {}
    for subdir, dirs, files in os.walk(path):
        for file in files:
            dct_of_files[os.path.join(subdir, file)] = os.path.join('', file)
    return dct_of_files

    
def get_files(dct, name):
    return {key:value for (key,value) in dct.items() if name in key}


def get_files_from_files(path, filters):
    dct = get_dirs(path)
    for name in filters:
        dct = get_files(dct, name)
    return dct    

    
def normalize(img):
    min_ = img.min()
    max_ = img.max()
    x = 2*(img - min_) / (max_ - min_) - 1
    return x    
    
    
def get_img_array(path):
    return normalize(np.array(tiff.imread(path)))
    
    
def get_img_mask_array(path):
    mask = np.array(tiff.imread(path)) / 65535.0
    return mask
        
    
def slice_data(data, h, w, step):
    data_w = data.shape[1]
    data_h = data.shape[0]
    return [data[step_h*step: h + step_h*step,  step_w*step: w + step_w*step] for step_h in range(data_h//step) for step_w in range(data_w//step) if h + step_h*step <= data_h and w + step_w*step <= data_w]    
    
    
def get_dct_with_names(dct, names):
    res = {}
    for i in names:
        for key in dct:
            if i in key:
                res[key] = dct[key]
    return res            