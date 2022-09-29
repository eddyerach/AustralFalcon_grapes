
![Logo](https://www.australfalcon.com/wp-content/uploads/2020/04/australfalcon-logo_83110dbd991fa114543f627f8df424f4.png)


# Deep learning for vineyard cultivation
Code for the entire development cycle of Deep Learning models focused on grape cultivation.
The implementation is based on instance segmentation and on Facebook's Detectron2 framework. It is able to detect bunches and count them and also to detect grapes and count them.
![](https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/Hilera_2_2_37_1_frame4320.jpg?raw=true | width=100)
![](https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/v1_dataset_60.jpg?raw=true )
![](https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/modelov3_aumento_color_th01_Hilera_2_2_37_2_SEG10.gif?raw=true )

# Grape bunch detector

### Requirements
### Training Dataset
301 bunch images labeled for training corresponding to the 9 datasets found.
### Validation Dataset
68 images for validation corresponding to the 9 datasets found.

### Hyperparameter
- Learning rate: 0.001
- 2000 iterations
- 301 imgs with 2593 labeled clusters
- Inference threshold 0.5
- Training time 27 minutes


# Grape Bunch Counter

Since a new detector was developed, a new bunch counting algorithm had to be developed as well. For bunch counting, Tracker DeepSort was used as it is compatible with Detectron2 *[link](https://github.com/sayef/detectron2-deepsort-pytorch)*. The tracking algorithm, allows to track an object in a sequence of frames (video), assigning it a unique identifier.

# Berry Detector

# Table of Content 

- [Installation](#Installation)
- [Data Configuration (images, labels)](#DataConfiguration)
- [Training Guide](#TrainingGuide)
- [Inference Guide](#InferenceGuide)
- [Result](#Result)
- [Metric Table](#MetricTable)

# Installation
First, we need to know that this project was carried out on a computer with the following features:
  - OS type: 64-bit  Ubuntu/Linux 18.04.6 LTS
  - Processor: 11th Gen Intel Core i7-11700 @2.50GHz x 16
  - Graphics: NVIDIA GeForce RTX 3060
  
 ### Docker commands
 
 Download image 
 ```
sudo docker pull eddyerach1/detectron2_banano_uvas:latest
```
Build a container from an image
```
sudo docker run --gpus all -it -v /home/grapes:/shared  --name detectron2_grapes
```
# Data Configuration (images, labels)  
### Dataset for training model
For training the model we use 9 datasets (see the table *[1](https://drive.google.com/drive/folders/1BJkxu0ZkTGP42Y71ELhT6vvvcNWSOSbG)* ). Images were labeled with **VGG Imagen Annotator software**  and using the polygon tool. This tool is important because it generate 3 points which allows us to draw through the image contour. 
|Dataset|Images with label |Total labels|
|-------|------------------|------------|
| Dataset 1 (Train & Val)  | 72  |  747 | 
| Dataset 2 (Train & Val)  | 34  |  36  |     
| Dataset 3 (Train & Val)  | 55  |  67  |
| Dataset 4 (Train & Val)  | 46  |  454 |
| Dataset 5 (Train & Val)  | 46  |  585 |
| Dataset 6 (Train & Val)  | 30  |  518 |
| Dataset 7 (Train & Val)  | 31  |  391 |
| Dataset 8 (Train & Val)  | 29  |  454 |
| Dataset 9 (Train & Val)  | 30  |  217 |

In addition, it is important to mention that when exporting the .json file with the labeled images a processing of the labels must be performed with the following script, where it takes as input the .json file, the source (images) and the folder where the files will be saved. 

# Training Guide

# Inference Guide

# Results

# Metric Table
