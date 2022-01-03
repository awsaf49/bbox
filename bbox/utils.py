from numba import jit
import cv2
import numpy as np
import random

__all__ = ['coco2yolo', 'yolo2coco', 'voc2coco', 'coco2voc', 'yolo2voc', 'voc2yolo',
           'bbox_iou', 'draw_bboxes', 'load_image']

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
    
    bboxes[..., 0] += bboxes[..., 2]/2
    bboxes[..., 1] += bboxes[..., 3]/2
    
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

@jit(nopython=True)
def bbox_iou(b1, b2):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    Args:
        b1 (np.ndarray): An ndarray containing N(x4) bounding boxes of shape (N, 4) in [xmin, ymin, xmax, ymax] format.
        b2 (np.ndarray): An ndarray containing M(x4) bounding boxes of shape (N, 4) in [xmin, ymin, xmax, ymax] format.

    Returns:
        np.ndarray: An ndarray containing the IoUs of shape (N, 1)
    """
#     0 = np.convert_to_tensor(0.0, b1.dtype)
    # b1 = b1.astype(np.float32)
    # b2 = b2.astype(np.float32)
    b1_xmin, b1_ymin, b1_xmax, b1_ymax = np.split(b1, 4, axis=-1)
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = np.split(b2, 4, axis=-1)
    b1_height = np.maximum(0, b1_ymax - b1_ymin)
    b1_width  = np.maximum(0, b1_xmax - b1_xmin)
    b2_height = np.maximum(0, b2_ymax - b2_ymin)
    b2_width  = np.maximum(0, b2_xmax - b2_xmin)
    b1_area = b1_height * b1_width
    b2_area = b2_height * b2_width

    intersect_xmin = np.maximum(b1_xmin, b2_xmin)
    intersect_ymin = np.maximum(b1_ymin, b2_ymin)
    intersect_xmax = np.minimum(b1_xmax, b2_xmax)
    intersect_ymax = np.minimum(b1_ymax, b2_ymax)
    intersect_height = np.maximum(0, intersect_ymax - intersect_ymin)
    intersect_width  = np.maximum(0, intersect_xmax - intersect_xmin)
    intersect_area   = intersect_height * intersect_width

    union_area = b1_area + b2_area - intersect_area
    iou = np.nan_to_num(intersect_area/union_area).squeeze()
    
    return iou

@jit(nopython=True)
def clip_bbox(bboxes_voc, height=720, width=1280):
    """Clip bounding boxes to image boundaries.

    Args:
        bboxes_voc (np.ndarray): bboxes in [xmin, ymin, xmax, ymax] format.
        height (int, optional): height of bbox. Defaults to 720.
        width (int, optional): width of bbox. Defaults to 1280.

    Returns:
        np.ndarray : clipped bboxes in [xmin, ymin, xmax, ymax] format.
    """
    bboxes_voc[..., 0::2] = bboxes_voc[..., 0::2].clip(0, width)
    bboxes_voc[..., 1::2] = bboxes_voc[..., 1::2].clip(0, height)
    return bboxes_voc

def str2annot(data):
    """Generate annotation from string.
    
    Args:
        data (str): string of annotation.
    
    Returns:
        np.ndarray: annotation in array format.
    """
    data  = data.replace('\n', ' ')
    data  = np.array(data.split(' '))
    annot = data.astype(float).reshape(-1, 5)
    return annot

def annot2str(data):
    """Generate string from annotation.
    
    Args:
        data (np.ndarray): annotation in array format.
    
    Returns:
        str: annotation in string format.
    """
    data   = data.astype(str)
    string = '\n'.join([' '.join(annot) for annot in data])
    return string

def load_image(image_path):
    return cv2.imread(image_path)[..., ::-1]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_bboxes(img, bboxes, classes, class_ids, colors = None, show_classes = None, bbox_format = 'yolo', class_name = False, line_thickness = 2):  
     
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255 ,0) if colors is None else colors
    
    if bbox_format == 'yolo':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes:
            
                x1 = round(float(bbox[0])*image.shape[1])
                y1 = round(float(bbox[1])*image.shape[0])
                w  = round(float(bbox[2])*image.shape[1]/2) #w/2 
                h  = round(float(bbox[3])*image.shape[0]/2)

                voc_bbox = (x1-w, y1-h, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(get_label(cls)),
                             line_thickness = line_thickness)
            
    elif bbox_format == 'coco':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes:            
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                w  = int(round(bbox[2]))
                h  = int(round(bbox[3]))

                voc_bbox = (x1, y1, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)

    elif bbox_format == 'voc':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes: 
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                x2 = int(round(bbox[2]))
                y2 = int(round(bbox[3]))
                voc_bbox = (x1, y1, x2, y2)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image
