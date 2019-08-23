import point_sumple

def get_prediction(model, sumple):
    return model.predict(sumple, verbose=1)

def get_origin_mask(imgMatrixRead, mask_path, point, sample_size):
    return point_sumple.get_sumple_of_point([mask_path], point, imgMatrixRead, sample_size)

def get_mask(pred, threshold):
    pred[pred < threshold] = 0.0
    pred[pred > 0.0] = 1.0
    return pred


def postproces(mask):
    pass
    
    
    