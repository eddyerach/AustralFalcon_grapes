
![Logo](https://www.australfalcon.com/wp-content/uploads/2020/04/australfalcon-logo_83110dbd991fa114543f627f8df424f4.png)


# Bunches Detector

Mask R-CNN with a ResNet-50 based on detectron2 
for bunches of grapes detection in vineyard cultivation images.

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/b10e6fbf1c3eb04b5163c515e4409ba0b083fbef/imgs/detector_bunch/inference.jpg" width="800"/> 
  <div class="caption" float="center" align="center"> Bunches detected in red vs Bunches labeled in blue.</div>
</p>


# Usage - Grapes detector

## Data preparation
The original dataset corresponds to 369 images of bunches of grapes 
mostly in rows of vineyard which contains 3015 bunches labels

### Datasets and labels
The images were labeled using the [VGG image annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html) 
tool from The University of Oxford using the polygon region shape 
tool. Once the labeling was finished, a python code was used to 
format the label JSON file in such a way that it is compatible 
with detectron2 for training.

### Data augmentation
Due to the different light conditions that images of bunches of grapes can present and the low amount of labels that are in the dataset, a color augmentation was
applied to the original images, specifically positive and negative variations of color filters were applied.

- Brightness (+/-)
- Contrast (+/-)
- Saturation (+/-)
- Gamma (+/-)
![]()

As a result, the dataset was augmented to 3304 images which were divided into 
two datasets 2701 images for train and 603 images for test in a proportion of 80% and 20% 
respectively.

## Training
Parameters and values used to train the object detector model were the following:

| Parameter             | Value         |
| :--------             | :-------      |
| `Learning rate`       | `0.001`       |
| `Iterations`          | `4000`        |
| `Train Images`        | `2701`      |
| `Nº Labels`           | `23233`      |


## Evaluation
The evaluation criteria used was Intersection over Union (IOU), an IOU score greater than 0.2 
between a detection and label is considered a true positive, 
otherwise it is a false positive and labels without detection 
were considered false negatives. In this way, the accuracy of the model is calculated through a confusion matrix.

| Nº Deteccions  | True positives      | False positives    | false negatives   | Accuracy   | Recall        |
| :--------      | :-------            | :------------      | :------------     | :----------| :------------ |
| `3704`         | `3344`              | `360`              | `819`             | `0.903`    | `0.803`       |

## Results

Mosaics of the evaluation through IOU of the Bunches detection model are shown below.

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/b10e6fbf1c3eb04b5163c515e4409ba0b083fbef/imgs/detector_bunch/test-False Negatives-2.png" width="800"/> 
  <div class="caption" float="center" align="center"> False Negatives detections.</div>
</p>

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/b10e6fbf1c3eb04b5163c515e4409ba0b083fbef/imgs/detector_bunch/test-False Negatives-2.png" width="800"/> 
  <div class="caption" float="center" align="center"> False Positives detections.</div>
</p>

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/e7671d5306f74fb24d754e47d11adabca7c32076/imgs/detector_bunch/test-True Positives-1.png" width="800"/> 
  <div class="caption" float="center" align="center"> True Positives, detections in red and labels in blue.</div>
</p>

## Color Reference

| Bounding Box      | Color                                                                |
| ----------------- | ------------------------------------------------------------------ |
| Label | ![#0a192f](https://via.placeholder.com/10/0000FF?text=+) | 
| Detection | ![#00b48a](https://via.placeholder.com/10/FF0000?text=+) |