# PLEASE USE THE APP EN SRC, all below is deprecated

# IAdet : The ultimate annotation tool for object detection

Object detection is all about bounding boxes. The IAdet tool enables users to teach the computer how to draw bounding boxes around a particular class of objects present in some images. As you annotate, a model is trained on the background and is used to provide predictions.

## Installation

1. Clone the repo
2. With Python 3.10 run:
  ```
  python -m venv env_iadet
  source env_iadet/bin/activate
  bash install.sh  # installing mmcv might take a long time
  ```

## Usage
```
bash launch.sh ANN_PATH
```

## Cite
If you find this project useful, cite our work:
```
@inproceedings{marchesoni2022iadet,
  title={IAdet: Human in the loop object detection},
  author={Marchesoni-Acland, Facciolo},
  booktitle={NeurIPS 2022 Workshop on Human in the Loop Learning},
  year={2022}
}
```


## How it works

1. The app depends on `annotations.json`, a file that has a list of bounding boxes for each filename. JSON schema (what we expect the elements to be):
```
{"filepath":"img_000123.jpg",
 "pred_bboxes":[/* may stay after annotation */],
 "ann_bboxes":[[left, top, right, bottom]],
 "state":"predicted"          // one of ["unseen", "predicted", "annotated"] 
}
 ```



The bounding boxes for one image are saved on this file whenever you leave the image. A model is trained on the background based on the annotations in this file. If prefetching an unlabeled image, the last checkpoint of the model is used to predict the bounding boxes for that image. These predictions are saved into `tmp.json` and are discarded when the annotations of an image are saved. 

2. when you annotate, there are 5 things you can do:
  - move to the next/previous image saving current annotations as correct: this means that **any time you leave the annotations should be correct**
  - remove all bounding boxes by pressing the `Remove All` button
  - add a bounding box by making two left clicks
  - remove a bounding box with one right click


