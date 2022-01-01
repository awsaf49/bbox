import pandas as pd
import numpy as np
from .utils import voc2coco, coco2voc, yolo2coco, coco2yolo, voc2yolo, yolo2voc
from tqdm import tqdm
import json
import argparse
import os
from loguru import logger

__all__ = ['Converter']

class NumpyEncoder(json.JSONEncoder):
    """ 
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    Special json encoder for numpy types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Converter(object):
    def __init__(self, 
                 df,
                 categories = [{'id': 0, 'name': 'cots'}],
                 format = 'coco'):
        self.df = df
        self.categories = categories
        self.format = format
        
    def convert(self, df):
        annotion_id = 0
        images = []
        annotations = []

        for i, row in tqdm(df.iterrows(), total = len(df)):

            images.append({
                "id": i,
                "file_name": f"{row['image_id']}", # image_id must be including extension(.jpg, .png, etc)
                "height": row.height,
                "width": row.width,
            })
            
            bboxes = row['annotations'].copy()
            bboxes = np.array(bboxes).astype(np.float32)
            if self.format!='voc':
                bboxes = eval(f"{self.format}2voc")(bboxes, row.height, row.width)
            bboxes[..., 0::2] = np.clip(bboxes[..., 0::2], 0, row.width) # x must be in [0, width]
            bboxes[..., 1::2] = np.clip(bboxes[..., 1::2], 0, row.height) # y must be in [0, height]
            bboxes = eval(f"voc2coco")(bboxes, row.height, row.width)
            bboxes = bboxes.astype(int).tolist()
            
            category_ids = row['category_ids'].copy() # category_id must be for each bbox
            
            for j, bbox in enumerate(bboxes):
                category_id = int(category_ids[j])
                annotations.append({
                    "id": annotion_id,
                    "image_id": i,
                    "category_id": category_id,
                    "bbox": list(bbox),
                    "area": int(bbox[2]) * int(bbox[3]),
                    "segmentation": [],
                    "iscrowd": 0
                })
                annotion_id += 1

        self.json_file = {'categories':self.categories, 'images':images, 'annotations':annotations}
        return self.json_file
    
    def save(self, json_file, path):
        with open(path, 'w') as f:
            json.dump(json_file, f, indent=4, cls=NumpyEncoder)
            
    def convert_save(self, path):
        json_file = self.convert(self.df)
        self.save(json_file, path)
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=str, default='../data/train.csv')
    parser.add_argument('--valid-csv', type=str, default='../data/valid.csv')
    parser.add_argument('--test-csv', type=str, default=None)
    parser.add_argument('--format', type=str, default='coco')
    parser.add_argument('--output-dir', type=str, default='../output')
    return parser.parse_args()
                        
@logger.catch                 
def main(opt):
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)
        
    train_df = pd.read_csv(opt.train_csv)
    train_converter = Converter(train_df, format=opt.format)
    train_converter.convert_save(f'{opt.output_dir}/annotations_train.json')
    
    if opt.valid_csv:
        valid_df = pd.read_csv(opt.valid_csv)
        valid_converter = Converter(valid_df, format=opt.format)
        valid_converter.convert_save(f'{opt.output_dir}/annotations_valid.json')
    
    if opt.test_csv:
        test_df = pd.read_csv(opt.test_csv)
        test_converter = Converter(test_df, format=opt.format)
        test_converter.convert_save(f'{opt.output_dir}/annotations_test.json')
    
if __name__=='__main__':
    opt = parse_opt()
    main(opt)