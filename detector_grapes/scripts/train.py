# python code for train detectron2 model detectorr

# import some common libraries
import numpy as np
import cv2
import os
import json
from os.path import exists

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
setup_logger()

# Set path to dataset, must contain folders "test" for validation and "train" for training
folder_name='path/to/dataset'

def get_dataset_dicts(directory):
    classes = ['disease']
    dataset_dicts = []
    for idx, filename in enumerate([file for file in os.listdir(directory) if file.endswith('.json')]):
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        filename = os.path.join(directory, img_anns["imagePath"])

        if not exists(filename):
            extension = filename.split('.')[-1]
            if extension == 'JPG' or extension == 'jpg':  
                filename = ('.').join(filename.split('.')[:-1]) + '.png'


        img = cv2.imread(filename)
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = img.shape[0]
        record["width"] = img.shape[1]
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

dicts = get_dataset_dicts(folder_name+'/')


for d in ["train", "test"]:
    
    DatasetCatalog.register("dataset_" + d, lambda d=d: get_dataset_dicts(folder_name+'/' + d))
    MetadataCatalog.get("dataset_" + d).set(thing_classes=['disease'])

dataset_metadata = MetadataCatalog.get("dataset_train")

# Set hyperparameters for training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 2000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

print(f'checkpoint dir: {cfg.OUTPUT_DIR}')
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)

trainer.train()