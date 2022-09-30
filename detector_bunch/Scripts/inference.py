# Set path to images to infer
path = '/path/to/folder/with/images_to_infer'  
# Set folder path where the inferred images are saved
path_result = r'/path/to/folder/where/images_are_saved' 

# The model path used for inference is set below

# import some common libraries
import numpy as np
import cv2
import os
import numpy
import pandas as pd
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

dataset = os.path.basename(path)

#load of weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = 'path/to/the_model/Grapes_Bunch_Detector_Augmented_Model.pth'     # Set path to the bunch detector model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TEST = ("dataset_test", )
cfg.TEST.DETECTIONS_PER_IMAGE = 200
predictor = DefaultPredictor(cfg)

#inference and plot of detections on images
font = cv2.FONT_HERSHEY_SIMPLEX
df = pd.DataFrame()
image_list = []
cdi_lis = []
cd = 0
for image in os.listdir(path):
    image_list.append(image)
    im = cv2.imread(os.path.join(path, image))
    outputs = predictor(im)
    pred_len = len(outputs["instances"])
    bbox_raw = outputs['instances'].to('cpu')
    bbox_raw = bbox_raw.get_fields()
    bbox_raw = bbox_raw['pred_boxes'].tensor.numpy()
    bbox_raw = list(map(numpy.ndarray.tolist, bbox_raw))
    bbox_raw = list(map(lambda x: list(map(int, x)), bbox_raw))#esta
    cd = cd + len(bbox_raw)
    cdi = len(bbox_raw)
    cdi_lis.append(cdi)

    scores_raw = outputs['instances'].to('cpu')
    scores_raw = scores_raw.get_fields()
    scores_raw = scores_raw['scores'].numpy()

    for bbox, score in zip(bbox_raw, scores_raw):
      left_top = tuple(bbox[:2])
      right_bottom = tuple(bbox[2:])
      score_height = (bbox[0], bbox[1] - 5) 
      im = cv2.rectangle(im,right_bottom,left_top,(0,0,255),2)
      im = cv2.putText(im,"{:.2f}".format(score),score_height, font, 0.3,(0,0,255),1,cv2.LINE_AA)
    
    os.chdir(path_result)
    cv2.imwrite(image, im.astype(np.float32))

print('CANTIDAD DE DETECCIONES:', cd)

df['detections'] = cdi_lis
df['name'] = image_list
df.set_index("name", inplace = True)
df.loc['total'] = df.sum(axis=0)
df.to_csv(dataset+'_inferred.csv')