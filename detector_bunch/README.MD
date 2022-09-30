

![Logo](https://www.australfalcon.com/wp-content/uploads/2020/04/australfalcon-logo_83110dbd991fa114543f627f8df424f4.png)


# Bunches Detector

Mask R-CNN with a ResNet-50 based on detectron2 
for bunches of grapes detection in vineyard cultivation images.

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/b10e6fbf1c3eb04b5163c515e4409ba0b083fbef/imgs/detector_bunch/inference.jpg" width="800"/> 
  <div class="caption" float="center" align="center"> Bunches detected in red vs Bunches labeled in blue.</div>
</p>


# Usage - Bunch detector

## Data preparation
The original dataset corresponds to 369 images of bunches of grapes 
mostly in rows of vineyard which contains 3015 bunches labels

### Datasets and labels
The images were labeled using the [VGG image annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html) 
tool from The University of Oxford using the polygon region shape 
tool. Once the labeling was finished, a python code was used to 
format the label JSON file in such a way that it is compatible 
with detectron2 for training.

#### Processing of VGG file to JSON files compatible with training
Run the script **process_label.py** from terminal

```http
  process_label.py --via_file path/to/VGG_labels_file.json --images_src path/to/labeled_dataset_images --images_dst path/to/dataset_folder_train_test

```

### Data augmentation
Due to the different light conditions that images of bunches of grapes can present and the low amount of labels that are in the dataset, a color augmentation was
applied to the original images, specifically positive and negative variations of color filters were applied.

- Brightness (+/-)
- Contrast (+/-)
- Saturation (+/-)
- Gamma (+/-)

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/e8971b6711a59c27a9a1b53c08961539d91014e6/imgs/detector_bunch/color filters.png" width="800"/> 
  <div class="caption" float="center" align="center"> Color filters applied to original image.</div>
</p>

As a result, the dataset was augmented to 3304 images which were divided into 
two datasets 2701 images for train and 603 images for test in a proportion of 80% and 20% 
respectively.

## Set docker container
Pull docker container from Docker Hub 
```http
  sudo docker pull eddyerach1/detectron2_banano_uvas:latest

```
Run docker container from terminal
```http
  sudo docker run --gpus all -it -v /home/grapes:/shared  --name detectron2_grapes

```

Start docker container from terminal
```http
  sudo docker start detectron2_grapes

```
Enter to docker container from terminal
```http
  sudo docker exec -it detectron2_grapes bash

```

## Training
Parameters and values used to train the object detector model were the following:

| Parameter             | Value         |
| :--------             | :-------      |
| `Learning rate`       | `0.001`       |
| `Iterations`          | `4000`        |
| `Train Images`        | `2701`        |
| `Nº Labels`           | `23233`       |
| `Training time`       | `42 minutes` |

#### Training from docker container
Configure the training parameters in **train.py** script according to your needs, then run the script from docker container terminal

```http
  python3 train.py

```

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


#### Image Inference
Go to the path inside docker where the script **inference.py** is located and run the script
```http
  python3 inference.py

```

## Appendix

The dataset provided is a sample of the original dataset used by Austral Falcon for model development.

Downloadable Files:

-[Grapes Bunch Detector Augmented Model](https://drive.google.com/drive/u/0/folders/1UDaHRN8NDshTNN3eioF40tQN8lJtrqsx)

-[Datasets](https://drive.google.com/drive/folders/1QsUUjNpMz2n27a6XmgzPKwRnUQ9z81HF)

