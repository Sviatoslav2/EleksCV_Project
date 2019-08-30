import random
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from segmentation_models import Linknet
from datetime import datetime

import tif_read
import point_sumple
import pointsOfFild



## Comput parameters
point_lat_lon = (55.498067, 9.906204)

main_path = os.getcwd()
number_paths = 3
img_size = 10980 
sample_size = 160
slice_step = 50
order_of_files = ['TIF_S2A_MSIL1C_20180515T103021_N0206_R108_T32UNG_20180515T124152.SAFE',
                  'TIF_S2A_MSIL1C_20180508T104031_N0206_R008_T32UNG_20180508T175127.SAFE',
                  'TIF_S2B_MSIL1C_20180530T103019_N0206_R108_T32UNG_20180530T123402.SAFE']
order_of_chanel = ['B02.tif','B03.tif','B04.tif','B05.tif'] 
number_of_chanels = len(order_of_chanel) * number_paths
mask_path = main_path + '\\Data\\' + 'out.tif'
number_chanels = number_paths * len(order_of_chanel)


def get_file_name():
    id = datetime.today().strftime('%Y%m') + datetime.today().strftime('_%Y%m%d%H%M%S') +'.csv'
    return "OutputData_" + id

# Exception classes
class DataException(Exception):
    pass
class IncorrectDataFiles(Exception):
    pass
class IncorrectMaskPath(Exception):
    pass
    

dct_data1 = tif_read.get_dirs(main_path+"\Data"+"\\"+ order_of_files[0] +"\\")
dct_data2 = tif_read.get_dirs(main_path+"\Data"+"\\"+ order_of_files[1] +"\\")
dct_data3 = tif_read.get_dirs(main_path+"\Data"+"\\"+ order_of_files[2] +"\\")


if dct_data1 == dct_data2 == dct_data3 == {}:
    raise DataException("Error while trying to collect paths! You should have ./Data in "+ main_path + ' and ' + str(['.\Data\\' + i for i in order_of_files]))

    
if not os.path.isfile(main_path+"\Data"+'\out.tif'):
    raise IncorrectMaskPath("Error while trying to collect paths! You should have \Data\\out.tif")

    
dct_data_mask = {main_path+"\Data"+'\out.tif':'out.tif' }
lst_path = point_sumple.get_channels_path(dct_data1, order_of_chanel) + point_sumple.get_channels_path(dct_data2, order_of_chanel) + point_sumple.get_channels_path(dct_data3, order_of_chanel)    


dct_data1 = tif_read.get_dct_with_names(dct_data1, order_of_chanel)
dct_data2 = tif_read.get_dct_with_names(dct_data2, order_of_chanel)
dct_data3 = tif_read.get_dct_with_names(dct_data3, order_of_chanel)


if len(dct_data1) != len(order_of_chanel) or len(dct_data2) != len(order_of_chanel) or len(dct_data3) != len(order_of_chanel):
    raise IncorrectDataFiles("Error while trying to collect paths!  You should have " + str(order_of_chanel) + ' in eche file in ./Data file')
    
# Load model    
model = Linknet('resnet50', classes=1, activation='sigmoid', input_shape=(sample_size, sample_size, number_chanels),encoder_weights=None)
Path_to_weights = 'Model\\model_segmentation.h5'
model.load_weights(Path_to_weights)


path = lst_path[0]
imgMatrixRead  = point_sumple.ImgMatrixRead(lst_path, mask_path)
imgcord = point_sumple.Imgcord(imgMatrixRead)




def random_generator(number_sample, img_size, sample_size):
    lst = [i for i in range(sample_size,img_size, sample_size)][0:number_sample]
    random.shuffle(lst)
    return lst

def IOU(lst_mask, lst_pred):
    if len(set(lst_mask).union(set(lst_pred))) == 0:
        return 0
    return len(set(lst_mask).intersection(set(lst_pred)))/len(set(lst_mask).union(set(lst_pred)))

metric = 0
points1 = [(55.497306, 9.885297), (55.499263, 9.885211), (55.494783, 9.879259), (55.498290, 9.872779), (55.508705, 9.862522), (55.509167, 9.866996), (55.506098, 9.852576), (55.504385, 9.846686), (55.504087, 9.841118)]
points2 = [(55.504206, 9.837540), (55.504470, 9.838838), (55.504084, 9.833071), (55.505141, 9.824123), (55.504473, 9.821613), (55.503197, 9.807301), (55.505044, 9.797859), (55.512092, 9.801700), (55.510725, 9.791915), (55.511284, 9.787860)]
points3 = [(55.512322, 9.784448), (55.514333, 9.785403), (55.519107, 9.790521), (55.518846, 9.796100), (55.519715, 9.794265), (55.519964, 9.784459), (55.461250, 9.847555), (55.460861, 9.853134), (55.458324, 9.8610570)] 
points = points1 + points2 + points3 
for point_lat_lon in points:
    #point_lat_lon = imgcord.cord_to_lat_lon(point[0], point[1])
    lst_mask = pointsOfFild.get_points_of_fild(point_lat_lon, imgMatrixRead, imgcord, model, sample_size, True)
    lst_pred = pointsOfFild.get_points_of_fild(point_lat_lon, imgMatrixRead, imgcord, model, sample_size, False)
    metric += IOU(lst_mask, lst_pred)
metric = metric / len(points)
print("IOU_my_points == ", metric)
#########################################################
metric = 0
lst = random_generator(5, img_size, sample_size)
points = [(i,j) for i in lst for j in lst]
key = 0
for point in points:
    key+=1
    print("number == ", key)
    point_lat_lon = imgcord.cord_to_lat_lon(point[0], point[1])
    lst_mask = pointsOfFild.get_points_of_fild(point_lat_lon, imgMatrixRead, imgcord, model, sample_size, True)
    lst_pred = pointsOfFild.get_points_of_fild(point_lat_lon, imgMatrixRead, imgcord, model, sample_size, False)
    metric += IOU(lst_mask, lst_pred)
metric = metric / len(points)
print("IOU_random_points == ", metric)    