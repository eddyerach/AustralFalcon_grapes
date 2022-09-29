
![Logo](https://www.australfalcon.com/wp-content/uploads/2020/04/australfalcon-logo_83110dbd991fa114543f627f8df424f4.png)


# Bunches Detector

Mask R-CNN with a ResNet-50 based on detectron2 
for bunches of grapes detection in vineyard cultivation images.



![](https://github.com/eddyerach/AustralFalcon_grapes/blob/b22c49ada2b7705c93f53b71f25f7592abc45cad/imgs/Hilera_2_2_37_1_frame4320.jpg)
# Usage - Grapes detector

## Data preparation
The used dataset corresponds to 369 images of bunches of grapes 
mostly in rows of vineyard which were divided into 
two datasets 301 images for train and 68 images for test in a proportion of 80% and 20% 
respectively.

### Datasets and labels
The images were labeled using the [VGG image annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html) 
tool from The University of Oxford using the polygon region shape 
tool. Once the labeling was finished, a python code was used to 
format the label JSON file in such a way that it is compatible 
with detectron2 for training.

### Data augmentation
Due to the different light conditions that can occur when 
capturing the images of bunches of grapes, a color increase was
made to the original images of the dataset, specifically positive and negative variations of color filters were applied.

- Brightness (+/-)
- Contrast (+/-)
- Saturation (+/-)
- Gamma (+/-)
![]()

## Training
Parameters and values used to train the object detector model were the following:

| Parameter             | Value         |
| :--------             | :-------      |
| `Learning rate`       | `0.001`       |
| `Iterations`          | `2000`        |
| `Train Images`        | `301`      |
| `Nº Labels`           | `2593`      |


## Evaluation
The evaluation criteria used was Intersection over Union (IOU), an IOU score greater than 0.2 
between a detection and label is considered a true positive, 
otherwise it is a false positive and labels without detection 
were considered false negatives. In this way, the accuracy of the model is calculated through a confusion matrix.
| Nº Deteccions  | True positives      | False positives    | false negatives   | Accuracy   | Recall        |
| :--------      | :-------            | :------------      | :------------     | :----------| :------------ |
| `669`          | `543`               | `126`              | `125`             | `0.812`    | `0.813`       |

## Results
