
![Logo](https://www.australfalcon.com/wp-content/uploads/2020/04/australfalcon-logo_83110dbd991fa114543f627f8df424f4.png)


# Grapes Detector

Mask R-CNN with a ResNet-50 grape detector based on detectron2 for berries detection in bunches of grapes images, allows the detection of circular, oval and elongated grapes shapes.

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/d04036c2d70b040cee3f6a789eec9efe9dd17c41/imgs/brunch_inference.jpeg" width="800"      height="700"/> 
  <div class="caption" float="center" align="center">  </div>
</p>

# Usage - Grapes detector
## Datasets and labels

The images were labeled using the [VGG image annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html) from The University of Oxford. Due to the different shapes that berries can present, two region shape tools were used, for round and oval berries the elliptical region shape, and for elongated berries the polygon region shape. Once the labeling was finished, a python code was used to format the label JSON file in such a way that it is compatible with detectron2 for training. 

### Data Original 

There exists an original dataset containing 274 images of grape clusters of different types of berries such as round, oval and elongated with a total of 17822 labels.
The original data were divided into two datasets, training and test, in a proportion of 80% and 20% respectively. A total of 219 images were taken for training and 55 images for validation.

### Data Augmentation

To achieve a robust model, we improved the accuracy of the rotated images with different types of berries. In addition, the number of images of grape clusters was increased from 274 to 5,054 with a total of 324,368 labels, thus dividing this dataset into training and test, in a proportion of 80% and 20% respectively. A total of 4017 images were taken for training and 1036 images for validation.

## Training

Two different models were trained, one with the original data set and the other with the augmented data set. In the first table you can see the parameters and values used to train the model with the original data set. In this case, 219 images of grape clusters were taken with a total of 17,607 labels, which took 56 minutes of training.  

*Table1:  Data Original*

| Parameter             | Value         |
| :--------             | :-------      |
| `Learning rate`       | `0,001`       |
| `Iterations`          | `2,000`        |
| `Train Images`        | `219`         |
| `Nº Labels`           | `17,607`       |
| `Time Training`       | `56 min`      |     

*Table2: Data Augmented*

| Parameter             | Value         |
| :--------             | :-------      |
| `Learning rate`       | `0,001`       |
| `Iterations`          | `6,000`        |
| `Train Images`        | `5,054`        |
| `Nº Labels`           | `324,368`      |
| `Time Training`       | `3h`          |

The rotations were from 45º to -45º, each 5º a rotation was made for a total of 19 rotations of the dataset.
<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/d04036c2d70b040cee3f6a789eec9efe9dd17c41/imgs/grape_detector_rotations.png" width="800"      height="600"/> 
  <div class="caption" float="center" align="center">  </div>
</p>

## Evaluation
### Data Original
Due to a large number of berries per bunch, it was decided to contrast the number of labels vs. the number of detections, since there are labels and detections that overlap. The evaluation of the model was carried out through the inference of the test dataset.

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/grapes_detection_original.png" width="800"      height="600"/> 
  <div class="caption" float="center" align="center">  </div>
</p>


|                               | Detection      | Regression    |
| :--------                     | :-------       | :------------ |
| `Average individual Accuracy` |  `0,90`        |  `0,90`       |
| `Totals Accuracy`             |  `0,91`        |  `0,97`       |

### Data Augmentation


<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/Grapes_Detection_augmentation.png" width="800" height="600"/> 
  <div class="caption" float="center" align="center">  </div>
</p>


|                               | Detection      | Regression    |
| :--------                     | :-------       | :------------ |
| `Average individual Accuracy` |  `0,92`        |  `0,92`       |
| `Totals Accuracy`             |  `0,93`        |  `0,99`       |

## Results
