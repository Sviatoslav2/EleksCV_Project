import point_sumple
import main
def get_prediction(model, sumple):
    return model.predict(sumple, verbose=1)

def get_origin_mask(imgMatrixRead, mask_path, point, sample_size):
    return point_sumple.get_sumple_of_point([mask_path], point, imgMatrixRead, sample_size)

def get_mask(pred, threshold):
    pred[pred < threshold] = 0.0
    pred[pred > 0.0] = 1.0
    return pred

def limits(mask):
    sample_size = mask.shape[0]
    for i in range(sample_size):
        mask[1][i] = 1.0
        mask[i][1] = 1.0
        
        mask[0][i] = 1.0
        mask[i][0] = 1.0
        
        mask[sample_size - 1][i] = 1.0
        mask[i][sample_size - 1] = 1.0
        
    return mask    

def postproces(mask):
    return get_mask(limits(mask), main.probability_thresh)
    
    
    