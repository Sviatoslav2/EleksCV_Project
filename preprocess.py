import numpy as np
import pandas as pd
import random

def conect_one_img_channels(lst_channels, idx):
    return [i[idx] for i in lst_channels]    
    
def get_multi_channel_img(lst_channels, idx):
    multi_channel_img = np.zeros((lst_channels[0][0].shape[0], lst_channels[0][0].shape[1], len(lst_channels)))
    one_img_channels = conect_one_img_channels(lst_channels, idx)
    for index, channel in enumerate(one_img_channels):
        multi_channel_img[:,:,index] = channel 
    
    return multi_channel_img

def get_samples(lst_channels, number1, number2):
    return [get_multi_channel_img(lst_channels, index) for index in range(number1, number2)]
    
def get_masks(dct_masks):
    return [dct_masks[key] for key in dct_masks][0]    
    
def get_channels(dct_tif, order):
    return [dct_tif[path] for name in order for path in dct_tif if name in path]
    
def masks_thensor(masks):
    res = []    
    for mask in masks:   
        one_channel_img = np.zeros((masks[0].shape[0], masks[0].shape[1], 1))
        one_channel_img[:,:,0] = mask
        res.append(one_channel_img)
    return res  

def get_one_masks_thensor(masks, index):
    one_channel_img = np.zeros((masks[0].shape[0], masks[0].shape[1], 1))
    one_channel_img[:,:,0] = masks[index]
    return one_channel_img
    
def generator(lst_channels, masks, batch_size, lim1, lim2):
    while True:
        samples = []
        labels = []
        for _ in range(batch_size): 
            index = random.randint(lim1,lim2)
            samples.append(get_multi_channel_img(lst_channels, index))
            labels.append(get_one_masks_thensor(masks, index))
        yield np.array(samples), np.array(labels)    
        #yield get_multi_channel_img(lst_channels, index), get_one_masks_thensor(masks, index)
    
