import point_sumple
import postprocess_modul
import sys
import numpy as np
import main


def get_origin_mask(imgMatrixRead, mask_path, point, sample_size):
    return point_sumple.get_sumple_of_point([mask_path], point, imgMatrixRead, sample_size)

def is_limit(point, sumple_size):
    return point[0] == sumple_size - 1 or point[1] == sumple_size - 1 or point[0] == 0 or point[1] == 0
    
    
def get_glob_cord_pix(imgcord, point_lat_lon):
    res_point = imgcord.latlon_to_pix(point_lat_lon[0],point_lat_lon[1])
    return res_point[0], res_point[1]

def get_lat_lon(imgcord, point_glob_pix):
    return imgcord.cord_to_lat_lon(point_glob_pix[0], point_glob_pix[1])

def get_pred_mask(point_glob_pix, imgMatrixRead, model, sample_size, present=False):
    if present:
        return postprocess_modul.limits(get_origin_mask(imgMatrixRead, imgMatrixRead.mask_path, point_glob_pix, sample_size))
    else:
        lst_path = imgMatrixRead.lst_path
        sumple = point_sumple.get_sumple_of_point(lst_path,point_glob_pix,imgMatrixRead, sample_size)
        return postprocess_modul.postproces(postprocess_modul.get_prediction(model, np.array([sumple]))[0])
    
def new_mask(point_glob_pix, imgMatrixRead, imgcord, model, sumple_size, present):
    imgcord = imgcord.modif_limits(point_glob_pix, sumple_size)
    return get_pred_mask(point_glob_pix, imgMatrixRead, model, sumple_size, present), imgcord  




def is_brake(mask, point, thresh):
    #key = 0
    for i in range(thresh):
        key1 = 0
        key2 = 0
        if point[1] + i < mask.shape[0] and mask[point[0]][point[1] + i] == 1.0:
            key1 += 1
        if point[1] - i < mask.shape[0] and mask[point[0]][point[1] - i] == 1.0:    
            key2 += 1
    if key1+key2 >= 2 and key1 and key2:
        return True
    for i in range(thresh):
        key1 = 0
        key2 = 0
        if point[0] + i < mask.shape[0] and mask[point[0] + i][point[1]] == 1.0:
            key1 += 1
        if point[0] - i < mask.shape[0] and mask[point[0] - i][point[1]] == 1.0:    
            key2 += 1
    if key1+key2 >= 2 and key1 and key2:
        return True
    return False        
    
    
def point_of_fild(mask, point_glob_pix, imgMatrixRead, imgcord, sumple_size, array_res):
    '''recursive_point3'''
    
    point_loc_pix = imgcord.get_sumple_pixels(point_glob_pix, sumple_size)
    stack = [point_loc_pix]
    
    def add_point_to_stack(point, array_res, mask, stack):
        if (not mask[point[0]][point[1]]) and (not array_res[point[0]][point[1]]) and (not is_brake(mask, point, main.border_thresh)):
            array_res[point[0]][point[1]] = 1.0
            stack.append(point)
        return stack, array_res
                    
    def points_around_point(point, array_res, mask, stack):
        if not array_res[point[0]][point[1]+1]:
            stack, array_res = add_point_to_stack((point[0],point[1]+1), array_res, mask, stack)
            
        if not array_res[point[0]][point[1]-1]:    
            stack, array_res = add_point_to_stack((point[0],point[1]-1), array_res, mask, stack)
            
        if not array_res[point[0]+1][point[1]]:
            stack, array_res = add_point_to_stack((point[0]+1,point[1]), array_res, mask, stack)
        
        if not array_res[point[0]-1][point[1]]:
            stack, array_res = add_point_to_stack((point[0]-1,point[1]), array_res, mask, stack)
        
        return stack, array_res
        
    while len(stack) != 0:
        stack, array_res = points_around_point(stack[0], array_res, mask, stack)
        stack.remove(stack[0])
        
    return [imgcord.get_global_pixels((i,j), sumple_size) for i in range(sumple_size) for j in range(sumple_size) if array_res[i][j]]    
    
    
    
def get_pix_local(lst_lat_lon, imgcord, sumple_size):
    def to_pix(point):
        return get_glob_cord_pix(imgcord, point)#imgcord.get_sumple_pixels(point, sumple_size)
        
    def to_loc_pix(point):
        return imgcord.get_sumple_pixels(to_pix(point), sumple_size)
        
    return list(map(to_loc_pix, lst_lat_lon))
    
    
def get_points_of_fild(point_lat_lon, imgMatrixRead, imgcord, model, sample_size, present):
    array_res = np.zeros((sample_size,sample_size))
    '''
        main function
    '''
    def to_lat_lon(point):
        return get_lat_lon(imgcord, point)
        
    point_glob_pix = get_glob_cord_pix(imgcord, point_lat_lon)
    pred, imgcord = new_mask(point_glob_pix, imgMatrixRead, imgcord, model, sample_size, present)
    pred = pred[:,:,0]
    # break line ! TODO
    return list(map(to_lat_lon, point_of_fild(pred, point_glob_pix, imgMatrixRead, imgcord, sample_size, array_res))) 
    