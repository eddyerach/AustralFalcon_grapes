

<p float="center" align="left">
  <img src="https://www.australfalcon.com/wp-content/uploads/2020/04/australfalcon-logo_83110dbd991fa114543f627f8df424f4.png"/>
  <h1> Austral Falcon: Deep learning for vineyard cultivation </h1>
</p>

Austral Falcon provides a technological solution designed to digitize the processes that are currently carried out manually in the agricultural industry. Our platform applies state-of-the-art Machine Vision (MV), Deep Learning (DL) and Machine Learning (ML) technology to optimize decision-making based on quantitative information, acquired systematically and with high precision.

<p align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/modelov3_aumento_color_th01_Hilera_2_2_37_2_SEG10_small.gif" alt="Counting grapes GIF"/>
  <div class="caption" float="center" align="center">Bunch Counting</div>
</p>


# Table of Content 

- [Web Platform](#web-platform)
- [Grapes Detector](#grapes-detector)
- [Bunch Detector](#bunch-detector)
- [Bunch Counter](#bunch-counter)

# Web Platform

We provide a platform with the ability to organize all the relevant information on your crops and extract useful data that allows you to plan and make more informed decisions. This component is better described [here](https://github.com/eddyerach/AustralFalcon_grapes/tree/main/web_platform).

<p align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/web_platform/imgs/img_multiespectral.png" width="1280" alt="Platform Login"/>
</p>


# Grapes Detector
Two Deep Learning models to detect two different variantes of grapes. These models were trained with rotation augmentations to guarantee its robustness.

This component is better described [here](https://github.com/eddyerach/AustralFalcon_grapes/tree/main/detector_grapes).

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/v1_dataset_60.jpg" height="360"/> 
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/bayas_largas.jpg" height="360"/> 
  <div class="caption" float="center" align="center">Grapes Detection on Two Variants</div>
</p>

The number of grapes counted in each cluster is adjusted by regression, which allows greater precision.
<p align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/grapes_detection_and_regression.png" width="640" alt="Grape counting prec"/>
</p>


# Bunch Detector

Deep Learning model with the ability to detect grape clusters in plants. The detection is performed using instance segmentation through a mask r-cnn neural network. It was implemented using the Meta Detectron2 framework and different data augmentation techniques were used to guarantee a robust model capable of offering good detection accuracy in different image or video capture conditions.

This component is better detailed [here](https://github.com/eddyerach/AustralFalcon_grapes/tree/main/detector_bunch).

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/Hilera_2_2_37_1_frame4320.jpg" width="640" height"360"/>
  <div class="caption" float="center" align="center">Bunch and Grapes Detection</div>
</p>


# Bunch Counter

The cluster counter is a Deep Learning system that takes the detections of the cluster detector and feeds it into a tracking algorithm, thus allowing the clusters to be identified in a video and then counted.

The tracker used is a variant of DeepSort and the counting algorithm consists of a heuristic based on line crossings. More detail can be found [here](https://github.com/eddyerach/AustralFalcon_grapes/tree/main/counter_grapes).

<p align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/modelov3_aumento_color_th01_Hilera_2_2_37_2_SEG10_small.gif" alt="Counting grapes GIF"/>
  <div class="caption" float="center" align="center">Bunch Counting</div>
</p>
