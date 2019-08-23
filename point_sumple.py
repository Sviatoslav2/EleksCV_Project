from osgeo import gdal
import geoio
import utm
import tif_read
import numpy as np
########################################### !
########################################### !
########################################### !

class ImgMatrixRead:
    def __init__(self, lst_path, mask_path):
        self.dct_path = {path: tif_read.get_img_array(path) for  path in lst_path}
        self.dct_path[mask_path] = tif_read.get_img_mask_array(mask_path)
        self.lst_path = lst_path
        self.mask_path = mask_path
        
    def get_matrix(self, path):
        return self.dct_path[path]
    
    def get_mask(self):
        return self.dct_path[self.mask_path]
########################################### !
########################################### !
########################################### !

def get_slice_img(path, point, imgMatrixRead ,sumple_size, key=False):
    
    imgcord = Imgcord(imgMatrixRead)
    img_matrix = imgMatrixRead.get_matrix(path)
    height_p, width_p = point[0], point[1]
    height_img, width_img = imgcord.get_size_file(path)
    
    h1 = height_p - sumple_size// 2 
    h2 = height_p + sumple_size // 2
    
    w1 = width_p - sumple_size // 2
    w2 = width_p  + sumple_size // 2
    
    if h1 < 0:
        h1 = 0
        h2 = sumple_size
    if h2 > height_img:
        h1 = height_img - sumple_size
        h2 = height_img
    if w1 < 0:
        w1 = 0
        w2 = sumple_size
    if w2 > width_img:
        w1 = width_img - sumple_size
        w2 = width_img  
    
    if not key:    
        return img_matrix[h1: h2, w1: w2]
    else: 
        return h1, h2, w1, w2
########################################### !
########################################### !
########################################### !

class Coordinate_exception(Exception):
    pass

class Imgcord:
    def __init__(self, imgMatrixRead):
        self.path = imgMatrixRead.lst_path[0]
        self.imgMatrixRead = imgMatrixRead
        self.GeoTransform = gdal.Open(imgMatrixRead.lst_path[0]).GetGeoTransform()
        self.limits = (0,0,0,0)
        
    def latlon_to_pix(self, lat, lon):
        x = utm.from_latlon(lat, lon)[0]
        y = utm.from_latlon(lat, lon)[1]
        img_width, img_height = self.get_size_file(self.path)
        width = (x - self.GeoTransform[0]) / self.GeoTransform[1]
        height = (y - self.GeoTransform[3]) / self.GeoTransform[5]
        if width < 0 or height < 0 or width > img_width or height > img_height:
            raise Coordinate_exception("Point is out of tiff image!")        
        return int(round(height)), int(round(width)) 
        
    def get_size_file(self, path):
        ds = gdal.Open(path)
        width = ds.RasterXSize
        height = ds.RasterYSize
        return height, width
    
    def cord_to_lat_lon(self, height, width):
        #print(self.GeoTransform)   (499980.0, 10.0, 0.0, 6200040.0, 0.0, -10.0)
        y = self.GeoTransform[3] + width *self.GeoTransform[4] + height *self.GeoTransform[5]
        x = self.GeoTransform[0] + width *self.GeoTransform[1] + height *self.GeoTransform[2]
        return utm.to_latlon(x, y, 32, 'U')
        #c, a, b, f, d, e = self.GeoTransform
        #xp = a * width + b * height + a * 0.5 + b * 0.5 + c
        #yp = d * width + e * height + d * 0.5 + e * 0.5 + f
        #return xp, yp

        
    
    def modif_limits(self, point, sumple_size):
        self.limits = get_slice_img(self.path, point, self.imgMatrixRead, sumple_size, True) 
        return self
        
    
    def get_sumple_pixels(self, point, sumple_size):
        h_glob = point[0]
        w_glob = point[1]
        h1, h2, w1, w2 = self.limits
        return  h_glob - h1, w_glob - w1
    
    def get_global_pixels(self, point, sumple_size):
        h_loc = point[0]
        w_loc = point[1]
        h1, h2, w1, w2 = self.limits
        return h_loc + h1, w_loc + w1    
        
########################################### !
########################################### !
########################################### !

def get_channels_path(dct_tif, order):
    return [path for name in order for path in dct_tif if name in path]        
        
def get_multi_img(lst_chanel,sumple_size):
    multi_channel_img = np.zeros((sumple_size, sumple_size, len(lst_chanel)))
    for index, channel in enumerate(lst_chanel):
        multi_channel_img[:,:,index] = channel
    return multi_channel_img 
    
def get_sumple_of_point(lst_path, point, imgMatrixRead, sumple_size):
    return get_multi_img([get_slice_img(path, point, imgMatrixRead, sumple_size) for path in lst_path], sumple_size)