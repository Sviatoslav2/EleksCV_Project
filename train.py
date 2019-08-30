import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from segmentation_models import Linknet


import tif_read
import preprocess



from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import *
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
import keras_metrics
from keras import backend

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint


def get_channels_path(dct_tif, order):
    return [path for name in order for path in dct_tif if name in path]      

# Comput parameters
main_path = os.getcwd()

number_paths = 3

img_size = 10980 

sample_size = 160

slice_step = 50

order_of_chanel = ['B02.tif','B03.tif','B04.tif','B05.tif'] 

number_of_chanels = len(order_of_chanel) * number_paths

sumpel = 20

batch_size = 55

num_epoch = 200

mask_path = main_path + '/Data/' + 'out.tif'
    
number_chanels = number_paths * len(order_of_chanel)

total_train = 0.8 # total / train

print("Comput parameters: Done!")

dct_data1 = tif_read.get_dirs(main_path+"/Data"+"/TIF_S2A_MSIL1C_20180515T103021_N0206_R108_T32UNG_20180515T124152.SAFE/")
dct_data2 = tif_read.get_dirs(main_path+"/Data"+"/TIF_S2A_MSIL1C_20180508T104031_N0206_R008_T32UNG_20180508T175127.SAFE/")
dct_data3 = tif_read.get_dirs(main_path+"/Data"+"/TIF_S2B_MSIL1C_20180530T103019_N0206_R108_T32UNG_20180530T123402.SAFE/")
dct_data_mask = {main_path+"/Data"+'/out.tif':'out.tif' }
lst_path = get_channels_path(dct_data1, order_of_chanel) + get_channels_path(dct_data2, order_of_chanel) + get_channels_path(dct_data3, order_of_chanel)

dct_data1 = tif_read.get_dct_with_names(dct_data1, order_of_chanel)
dct_data2 = tif_read.get_dct_with_names(dct_data2, order_of_chanel)
dct_data3 = tif_read.get_dct_with_names(dct_data3, order_of_chanel)

print("Collecting paths: Done!")

# Data preparing
channels1 = []
channels2 = []
channels3 = []


if number_paths not in [1, 2, 3]:
    raise Exception("number_paths must be == 1 or 2 or 3 !")

if number_paths >= 1:
    dct1 = {key:tif_read.slice_data(tif_read.get_img_array(key),sample_size,sample_size, slice_step) for (key,value) in dct_data1.items()}
    dct_mask = {key:tif_read.slice_data(tif_read.get_img_mask_array(key),sample_size,sample_size, slice_step) for (key,value) in dct_data_mask.items()}
    channels1 = preprocess.get_channels(dct1, order_of_chanel)
    del dct1
if number_paths >= 2:
    dct2 = {key:tif_read.slice_data(tif_read.get_img_array(key),sample_size,sample_size, slice_step) for (key,value) in dct_data2.items()}
    channels2 = preprocess.get_channels(dct2, order_of_chanel)
    del dct2
if number_paths >= 3:
    dct3 = {key:tif_read.slice_data(tif_read.get_img_array(key),sample_size,sample_size, slice_step) for (key,value) in dct_data3.items()}
    channels3 = preprocess.get_channels(dct3, order_of_chanel)
    del dct3
lst_channels = channels1 + channels2 + channels3 
number_sample = len(channels1[0])
del channels1
del channels2
del channels3
masks_lst = preprocess.get_masks(dct_mask)
del dct_mask
print("Data preparing: Done!")    


model = Linknet('resnet50', classes=1, activation='sigmoid', input_shape=(sample_size, sample_size, number_chanels),encoder_weights=None)
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy',keras_metrics.precision(), keras_metrics.recall()])
model.summary()    


callbacksList = [
    EarlyStopping(patience=10, verbose=1),
    ModelCheckpoint('model_segmentation.h5', verbose=1,
    save_best_only=True,
    save_weights_only=True,
    period=50) # Interval (number of epochs) between checkpoints. 
]

lim1_train, lim2_train, lim1_test, lim2_test = preprocess.data_train_test_index(number_sample, total_train)

train_generator = preprocess.generator(lst_channels, masks_lst, batch_size, lim1_train, lim2_train, number_sample)
validation_generator = preprocess.generator(lst_channels, masks_lst, batch_size, lim1_test, lim2_test, number_sample)


validation_steps = number_sample * ( 1 - total_train) // batch_size
samples_per_epoch = number_sample * total_train // batch_size


results = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=num_epoch,verbose=1,
                              callbacks=callbacksList) # Save the model
                              
print("Dane !!!")                                