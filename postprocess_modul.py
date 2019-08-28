import point_sumple

def get_prediction(model, sumple):
    return model.predict(sumple, verbose=1)

def get_origin_mask(imgMatrixRead, mask_path, point, sample_size):
    return point_sumple.get_sumple_of_point([mask_path], point, imgMatrixRead, sample_size)

def get_mask(pred, threshold):
    pred[pred < threshold] = 0.0
    pred[pred > 0.0] = 1.0
    return pred




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









def get_slice_matrix(mask, threshold, point, sumple_size):
    height_p, width_p = point[0], point[1]
    height_img = sumple_size
    width_img = sumple_size
    
    h1 = height_p - threshold// 2 
    h2 = height_p + threshold // 2
    
    w1 = width_p - threshold // 2
    w2 = width_p  + threshold // 2
    
    if h1 < 0:
        h1 = 0
        h2 = threshold
    if h2 > height_img:
        h1 = height_img - threshold
        h2 = height_img
    if w1 < 0:
        w1 = 0
        w2 = threshold
    if w2 > width_img:
        w1 = width_img - threshold
        w2 = width_img  
    
    return img_matrix[h1: h2, w1: w2]
    



def postproces(mask):
    return get_mask(mask, 0.7)
    
    
    