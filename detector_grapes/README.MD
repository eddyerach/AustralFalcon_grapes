
![Logo](https://www.australfalcon.com/wp-content/uploads/2020/04/australfalcon-logo_83110dbd991fa114543f627f8df424f4.png)


# Grapes Detector

Mask R-CNN with a ResNet-50 grape detector based on detectron2 
for berries detection in bunches of grapes images, allows the 
detection of circular, oval and elongated grapes shapes.

![](https://github.com/eddyerach/AustralFalcon_grapes/blob/d04036c2d70b040cee3f6a789eec9efe9dd17c41/imgs/brunch_inference.jpeg)

# Usage - Grapes detector

## Data preparation
The dataset used corresponds to 274 images of bunches of grapes which were divided into two datasets train and test in a proportion of 80% and 20% respectively.

### Datasets and labels
The images were labeled using the [VGG image annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html) 
tool from The University of Oxford. Due to the different shapes 
that berries can present, two region shape tools were used, 
for round and oval berries the elliptical region shape and for elongated 
berries the polygon region shape. Once the labeling was finished, a python code was used to format the label JSON file in such a way that it is compatible with detectron2 for training.

### Data augmentation
In order to have a higher accuracy of the model a retraining must be performed.  For this purpose, the number of images to be trained was increased, more variability of turns and training with different types of berries. The augmentation dataset corresponds to 5054 images of bunches of grapes with a total of 324368 labels. 

## Training
Parameters and values used to train the object detector model were the following:

| Parameter             | Value         |
| :--------             | :-------      |
| `Learning rate`       | `0.001`       |
| `Iterations`          | `2000`        |
| `Train Images`        | `274`         |
| `Nº Labels`           | `17607`       |

Parameters and values used to train the object detector model with data augmentation were the following:

| Parameter             | Value         |
| :--------             | :-------      |
| `Learning rate`       | `0.001`       |
| `Iterations`          | `6000`        |
| `Train Images`        | `5054`        |
| `Nº Labels`           | `324368`      |
| `Dataset train`       | `4017`        |
| `Dataset val`         | `1036`        |
| `Umbral test`         | `0.2`         |
| `Time Training`       | `3h`          |

The rotations were from 45º to -45º, each 5º a rotation was made for a total of 19 rotations of the dataset.

![](https://github.com/eddyerach/AustralFalcon_grapes/blob/d04036c2d70b040cee3f6a789eec9efe9dd17c41/imgs/grape_detector_rotations.png)

## Evaluation
Due to a large number of berries per bunch, it was decided to contrast the number of labels vs. the number of detections, since there are labels and detections that overlap. The evaluation of the model was carried out through the inference of the test dataset.

![](https://github.com/eddyerach/AustralFalcon_grapes/blob/d04036c2d70b040cee3f6a789eec9efe9dd17c41/imgs/grape_detector_chart.png)


|                               | Detection      | Regression    |
| :--------                     | :-------       | :------------ |
| `Average individual Accuracy` | `0.907`        | `0.902`       |
| `Totals Accuracy`             | `0.911`        | `0.971`       |

## Results
