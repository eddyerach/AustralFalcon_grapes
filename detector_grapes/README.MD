
![Logo](https://www.australfalcon.com/wp-content/uploads/2020/04/australfalcon-logo_83110dbd991fa114543f627f8df424f4.png)


# Grapes Detector

Mask R-CNN with a ResNet-50 grape detector based on detectron2 for berries detection in bunches of grapes images, allows the detection of circular, oval and elongated grapes shapes.

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/d04036c2d70b040cee3f6a789eec9efe9dd17c41/imgs/brunch_inference.jpeg" width="600"      height="600"/> 
  <div class="caption" float="center" align="center">  </div>
</p>

# Usage - Grapes detector
## Datasets and labels

The images were labeled using the [VGG image annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html) from The University of Oxford. Due to the different shapes that berries can present, two region shape tools were used, for round and oval berries the elliptical region shape, and for elongated berries the polygon region shape. Once the labeling was finished, a python code was used to format the label JSON file in such a way that it is compatible with detectron2 for training.

#### Processing of VGG file to JSON files compatible with training
Run the script **process_label.py** from terminal

```http
  process_label.py --via_file path/to/VGG_labels_file.json --images_src path/to/labeled_dataset_images --images_dst path/to/dataset_folder_train_test

```

### Data Original 

There exists an original dataset containing 274 images of grape buch of different types of berries such as round, oval and elongated.
The original data were divided into two datasets, training and test, in a proportion of 80% and 20% respectively. A total of 219 images were taken for training and 55 images for validation.

### Data Augmentation

To achieve a robust model, we improved the accuracy of the rotated images with different types of berries. In addition, the number of images of grape bunch was increased from 274 to 5,054 with a total of 324,368 labels, thus dividing this dataset into training and test, in a proportion of 80% and 20% respectively. A total of 4,017 images were taken for training and 1,036 images for validation.
The rotations were from 45º to -45º, each 5º a rotation was made for a total of 19 rotations of the dataset.

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/grape_detector_rotations.png" width="600"      height="600"/> 
  <div class="caption" float="center" align="center">  </div>
</p>

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

Two different models were trained, one with the original data set and one with the augmented data set. The parameters and values used for training are detailed below. 

### Training Data Original
*Table 1*

| Parameter             | Value         |
| :--------             | :-------      |
| `Learning rate`       | `0,001`       |
| `Iterations`          | `2,000`       |
| `Train Images`        | `219`         |
| `Nº Labels`           | `17,607`      |
| `Time Training`       | `56 min`      |     

### Training Data Augmented 
*Table 2*

| Parameter             | Value         |
| :--------             | :-------      |
| `Learning rate`       | `0,001`       |
| `Iterations`          | `6,000`       |
| `Train Images`        | `5,054`       |
| `Nº Labels`           | `324,368`     |
| `Time Training`       | `3 hours`     |


#### Training from docker container
Configure the training parameters in **train.py** script according to your needs, then run the script from docker container terminal

```http
  python3 train.py

```

## Evaluation

Due to a large number of berries per bunch, it was decided to contrast the number of labels vs. the number of detections, since there are labels and detections that overlap. The evaluation of the model was carried out through the inference of the test dataset.

### Data Original Model
<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/grapes_detection_original.png" width="600" height="600"/> 
  <div class="caption" float="center" align="center">  </div>
</p>


|                               | Detection      | Regression    |
| :--------                     | :-------       | :------------ |
| `Average individual Accuracy` |  `0,90`        |  `0,90`       |
| `Totals Accuracy`             |  `0,91`        |  `0,97`       |

### Data Augmentation Model

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/Grapes_Detection_augmentation.png" width="600" height="600"/> 
  <div class="caption" float="center" align="center">  </div>
</p>


|                               | Detection      | Regression    |
| :--------                     | :-------       | :------------ |
| `Average individual Accuracy` |  `0,92`        |  `0,92`       |
| `Totals Accuracy`             |  `0,93`        |  `0,99`       |

## Results

To conclude, it is observed how the augmented data model can identify different types of berries (round, oval, elongated). The following inferences are shown below. 

<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/elongated%20berries.png" width="600" height="600"/> 
  <div class="caption" float="center" align="center">   </div>
</p>


<p float="center" align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/round-and-oval-berries.png" width="600" height="600"/> 
  <div class="caption" float="center" align="center"> </div>
</p>


#### Image Inference
Go to the path inside docker where the script **inference.py** is located and run the script
```http
  python3 inference.py

```

## Appendix

Downloadable Files:

-[Grapes Bunch Detector Augmented Model](https://drive.google.com/drive/u/0/folders/1UDaHRN8NDshTNN3eioF40tQN8lJtrqsx)

-[Datasets](https://drive.google.com/drive/u/0/folders/1QsUUjNpMz2n27a6XmgzPKwRnUQ9z81HF)
