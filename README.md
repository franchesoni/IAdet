
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
bash launch.sh DATA_DIR
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
or check out the paper's code here: https://github.com/franchesoni/iadet_paper

## Future 
- Include semi-supervised learning https://mmdetection.readthedocs.io/en/v3.0.0rc0/user_guides/semi_det.html#configure-meanteacherhook
- Make GUI show the images in the center of the region (resize accroding to max width max depth)
