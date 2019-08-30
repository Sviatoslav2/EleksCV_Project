import os
import numpy as np
import pandas as pd
import tensorflow as tf
from segmentation_models import Linknet
from datetime import datetime





## Comput parameters
point_lat_lon = (55.497708, 9.911642)
probability_thresh = 0.7
border_thresh = 3 # radius around pixel 

order_of_files = ['TIF_S2A_MSIL1C_20180515T103021_N0206_R108_T32UNG_20180515T124152.SAFE',
                  'TIF_S2A_MSIL1C_20180508T104031_N0206_R008_T32UNG_20180508T175127.SAFE',
                  'TIF_S2B_MSIL1C_20180530T103019_N0206_R108_T32UNG_20180530T123402.SAFE']
order_of_chanel = ['B02.tif','B03.tif','B04.tif','B05.tif'] 


if __name__ == '__main__':
    
    import tif_read
    import point_sumple
    import pointsOfFild
    
    
    main_path = os.getcwd()
    img_size = 10980 
    number_paths = len(order_of_files)
    sample_size = 160
    slice_step = 50
    
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

    if number_paths >= 1:
        dct_data1 = tif_read.get_dct_with_names(dct_data1, order_of_chanel)
    else:
        dct_data1 = {}
    if number_paths >= 2:    
        dct_data2 = tif_read.get_dct_with_names(dct_data2, order_of_chanel)
    else:
        dct_data2 = {}
    if number_paths >= 3:
        dct_data3 = tif_read.get_dct_with_names(dct_data3, order_of_chanel)
    else:
        dct_data3 = {}


    if len(dct_data1) != 0 and len(dct_data1) != len(order_of_chanel):
        raise IncorrectDataFiles("Error while trying to collect paths!  You should have " + str(order_of_chanel) + ' in eche file in ./Data file')
    if len(dct_data2) != 0 and len(dct_data2) != len(order_of_chanel):
        raise IncorrectDataFiles("Error while trying to collect paths!  You should have " + str(order_of_chanel) + ' in eche file in ./Data file')
    if len(dct_data3) != 0 and len(dct_data3) != len(order_of_chanel):
        raise IncorrectDataFiles("Error while trying to collect paths!  You should have " + str(order_of_chanel) + ' in eche file in ./Data file')
        
    
    # Load model    
    model = Linknet('resnet50', classes=1, activation='sigmoid', input_shape=(sample_size, sample_size, number_chanels),encoder_weights=None)
    Path_to_weights = 'Model\\model_segmentation.h5'
    model.load_weights(Path_to_weights)

    # Load data 
    path = lst_path[0]
    imgMatrixRead  = point_sumple.ImgMatrixRead(lst_path, mask_path)
    imgcord = point_sumple.Imgcord(imgMatrixRead)

    def IOU(lst_mask, lst_pred):
        if len(set(lst_mask).union(set(lst_pred))) == 0:
            return 0
        return len(set(lst_mask).intersection(set(lst_pred)))/len(set(lst_mask).union(set(lst_pred)))

    lst_lat_lon = pointsOfFild.get_points_of_fild(point_lat_lon, imgMatrixRead, imgcord, model, sample_size, False)


    lst_lat_lon_ = pointsOfFild.get_points_of_fild(point_lat_lon, imgMatrixRead, imgcord, model, sample_size, True)
    print("IOU == ", IOU(lst_lat_lon_, lst_lat_lon))
    pd.DataFrame(data={'lat' : list(map(lambda x:x[0],lst_lat_lon)),'long':list(map(lambda x:x[1],lst_lat_lon))}).to_csv(get_file_name())