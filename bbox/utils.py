from numba import jit

@jit(nopython=True)
def voc2yolo(bboxes, height=720, width=1280):
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
#     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., 0::2] /= width
    bboxes[..., 1::2] /= height
    
    bboxes[..., 2] -= bboxes[..., 0]
    bboxes[..., 3] -= bboxes[..., 1]
    
    bboxes[..., 0] = bboxes[..., 0] + bboxes[..., 2]/2
    bboxes[..., 1] = bboxes[..., 1] + bboxes[..., 3]/2
    
    return bboxes

@jit(nopython=True)
def yolo2voc(bboxes, height=720, width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    
    """ 
#     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., 0::2] *= width
    bboxes[..., 1::2] *= height
    
    bboxes[..., 0:2] -= bboxes[..., 2:4]/2
    bboxes[..., 2:4] += bboxes[..., 0:2]
    
    return bboxes

@jit(nopython=True)
def coco2yolo(bboxes, height=720, width=1280):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
#     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # normolizinig
    bboxes[..., 0::2] /= width
    bboxes[..., 1::2] /= height
    
    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., 0:2] += bboxes[..., 2:4]/2
    
    return bboxes

@jit(nopython=True)
def yolo2coco(bboxes, height=720, width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]
    
    """ 
#     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # denormalizing
    bboxes[..., 0::2] *= width
    bboxes[..., 1::2] *= height
    
    # converstion (xmid, ymid) => (xmin, ymin) 
    bboxes[..., 0:2] -= bboxes[..., 2:4]/2
    
    return bboxes

@jit(nopython=True)
def voc2coco(bboxes, height=720, width=1280):
    """
    voc  => [xmin, ymin, xmax, ymax]
    coco => [xmin, ymin, w, h]
    
    """ 
#     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # converstion (xmax, ymax) => (w, h) 
    bboxes[..., 2:4] -= bboxes[..., 0:2]
    
    return bboxes

@jit(nopython=True)
def coco2voc(bboxes, height=720, width=1280):
    """
    coco => [xmin, ymin, w, h]
    voc  => [xmin, ymin, xmax, ymax]
    
    """ 
#     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # converstion (w, h) => (w, h) 
    bboxes[..., 2:4] += bboxes[..., 0:2]
    
    return bboxes
