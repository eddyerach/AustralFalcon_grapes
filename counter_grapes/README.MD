
![Logo](https://www.australfalcon.com/wp-content/uploads/2020/04/australfalcon-logo_83110dbd991fa114543f627f8df424f4.png)


# Grape bunch counter

The grape bunch counter is based on an [algorithm](https://github.com/sayef/detectron2-deepsort-pytorch) which allows tracking an object in a sequence of frames (video), each detected object is assigned a unique identifier which is maintained during the inference.



![](https://github.com/eddyerach/AustralFalcon_grapes/blob/b22c49ada2b7705c93f53b71f25f7592abc45cad/imgs/Hilera_2_2_37_1_frame4320.jpg)
# Counting Methods

## Crossing Lines
Four crossing lines were drawn on the video, the crossing lines 
are responsible for counting a bunch after it has been detected 
and an ID has been assigned. The count direction is from right to left, in the same direction in which new bunches are shown throughout the video.

![](https://lh3.googleusercontent.com/XGv-2ai-ualNYen6iEL0G8dWdXFqLZ6F6LW-kba9JV7j-GM09bTRFZ25GmmHOGbOLnD58iGePEnnckuLNGkSwpLuE4LglJIrpKz6M9Gn4cjddrLRUNoBAOqUDNrmdyZHLGi5cbeUBHdn-j4mosxtGSmhPC3f_EBMet8ziN1jFeRWRraK1Ktc__t8Duvn)

After a bunch is detected, the bunch bounding box must cross to the left side of the crossing line, then the ID which has been assigned by the tracker algorithm is recorded in a count list, the number of records stored in the list is displayed at the top right of each line. Finally, the total number of bunches counted is calculated through the average number of bunches counted by the four line crossings.


## Results

The accuracy of the cluster count depends on the accuracy of the detector model to correctly detect the clusters found in the video.

| Real  | LC 1     |LC 2 | LC 3     |LC 4  | Avg. LC  |Accu. Avg. | Dif Avg. Bunch  |
| :---- | :------- |:----| :------- |:---- | :------- |:--------  | :-------        |
| `799` | `694`    |`799`| `732`    |`580` | `701.25` |`0.87`     | `31`            |

<p align="center">
  <img src="https://github.com/eddyerach/AustralFalcon_grapes/blob/main/imgs/modelov3_aumento_color_th01_Hilera_2_2_37_2_SEG10_small.gif" alt="Counting grapes GIF"/>
  <div class="caption" float="center" align="center">Bunch Counting</div>
</p>