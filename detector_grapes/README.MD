
![Logo](https://www.australfalcon.com/wp-content/uploads/2020/04/australfalcon-logo_83110dbd991fa114543f627f8df424f4.png)


# Grapes Detector

Mask R-CNN with a ResNet-50 grape detector based on detectron2 
for berries detection in bunches of grapes images, allows the 
detection of circular, oval and elongated grapes shapes.

![](C:\Users\Dell\Documents\GitHub\AustralFalcon_grapes\imgs\brunch_inference.jpeg)

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
The rotations were from 45º to -45º, each 5º a rotation was made for a total of 19 rotations of the dataset.

![](imgs\grape_detector_rotations.png)

## Training
Parameters and values used to train the object detector model were the following:

| Parameter             | Value         |
| :--------             | :-------      |
| `Learning rate`       | `0.001`       |
| `Iterations`          | `2000`        |
| `Train Images`        | `219`      |
| `Nº Labels`           | `17607`      |


## Evaluation
Due to the large number of berries per bunch, it was decided to contrast the number of labels vs. the number of detections, since there are labels and detections that overlap.

![](imgs\grape_detector_chart.png)


|                               | Detection      | Regression    |
| :--------                     | :-------       | :------------ |
| `Average individual Accuracy` | `0.907`        | `0.902`       |
| `Totals Accuracy`             | `0.911`        | `0.971`       |

## Results