
# IAdet : The ultimate annotation tool for object detection

Object detection is all about bounding boxes. The IAdet tool enables users to teach the computer how to draw bounding boxes around a particular class of objects present in some images. As you annotate, a model is trained on the background and is used to provide predictions.

## Installation

python -m venv env_iadet
source env_iadet/bin/activate
bash install.sh

1. Clone the repo
2. `pip install -r requirements.txt` in your environment with Python 3.10
3. `python IAdet.py`

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
