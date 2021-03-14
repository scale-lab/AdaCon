# AdaCon
Adaptive Context-Aware Object Detection for Resource-Constrained Embedded Devices

## Abstract 
Convolutional Neural Networks achieve state-of-the-art accuracy in object detection tasks. However, they have large compute and memory footprints that challenge their deployment on resource-constrained edge devices. In AdaCon, we leverage the prior knowledge about the probabilities that different object categories can occur jointly to increase the efficiency of object detection models. In particular, our technique clusters the object categories based on their spatial co-occurrence probability. We use those clusters to design a hierarchical adaptive network. During runtime, a branch controller chooses which part(s) of the network to execute based on the spatial context of the input frame. 

## Paper
Will be available soon.

## How It Works
AdaCon consists of two main steps:

### Spatial-Context based Clustering 

<img src="doc/imgs/clustering_spatial_context.jpg" width="90%" style="display: block;  margin: 0 auto;">

Spatial-context based Clustering has 5 steps. First, we construct the co-occurrence matrix of the object categories where each value represents the frequency of the co-occurrence of the object categories in the same scene across all the training dataset. Then, we remove the common objects. Next, we convert the frequency-based co-occurrence matrix to a correlation matrix. Then, we use the correlation matrix to build a knowledge graph using Fruchterman-Reingold force-directed algorithm. Finally, we cluster the objects based on their location in the knowledge graph


### Adaptive Object Detction model
<img src="doc/imgs/adaptive_architecture.jpg" width="90%" style="display: block;  margin: 0 auto;">

Our adaptive object detection model consists of three components: a **backbone**, a **branch controller**, and a **pool of specialized detection heads** (branches). The backbone is first executed to extract the common features in the input image, then a branch controller takes the extracted features and route them towards one or more of the downstream specialized branches. Only the chosen branch(es) are then executed to get the detected object categories and their bounding boxes. During runtime, it has two modes of operation: **single-branch execution mode** where only the branch with the highest confidence score gets executed, and **multi-branch execution mode** where all the branches with a confidence score higher than a certain threshold are executed.


## Requirements

1. Clone the repo `git clone https://github.com/scale-lab/AdaCon.git ; cd AdaCon`

2. Create a virtual environment with Python 3.7 or later `python -m venv env ; source env/bin/activate`

3. Install the requirements using `pip install -r requirements.txt`

## Quick Start Demo

1. Download the pretrained AdaCon Model `./weights/download_adacon_weights.sh`

2. **Optional** Download the pretrained YOLO models. Run `./weights/download_yolov3_weights.sh`

3. Run Inference using `python detect.py --model model.args --adaptive --source 0`

## Inference 
detect.py runs inference on a variety of sources, and save the results to `runs/detect`
```
python detect.py --model {MODEL.args}
                 --source {SRC}
                 [--adaptive]
                 [--single]
                 [--multi]
                 [--bc-thres {THRES}]
                 [--img-size {IMG_SIZE}]
```
- `MODEL.args` is the path of the desired model description, check [model.args](https://github.com/scale-lab/AdaCon/blob/master/model.args) for an example.
- `SRC` is the input source. Set it to:
  - `0` for webcam.
  - `file.jpg` for image.
  - `file.mp4` for video.
  - `path` for directory.
- `adaptive` flag enables running the adaptive AdaCon model, otherwise the static model is executed.
- `single` flag enables single-branch execution of AdaCon.
- `multi` flag running multi-branch execution of AdaCon (Default).
- `THRES` is the confidence threshold for AdaCon's branch controller (Default = 0.4).
- `IMG_SIZE` is the resolution of the input image.

## Prepare your data for Testing and/or Training 
- To download COCO dataset, run `cd data ; ./get_coco2014.sh` or `cd data ; ./get_coco2017.sh` to download COCO 2014 or COCO 2017 respectively. 
- To download your custom dataset, follow this [tutorial](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data).

## Testing
test.py runs test on different datasets.
```
python test.py --model {MODEL.args}
                 --data {DATA.data}
                 [--adaptive]
                 [--single]
                 [--multi]
                 [--bc-thres {THRES}]
                 [--img-size {IMG_SIZE}]
```
- `MODEL.args` is the path of the desired model description, check [model.args](https://github.com/scale-lab/AdaCon/blob/master/model.args) for an example.
- `DATA.data` is the path of the desired data, use [data/coco2014.data](https://github.com/scale-lab/AdaCon/blob/master/data/coco2014.data) and [data/coco2017.data](https://github.com/scale-lab/AdaCon/blob/master/data/coco2017.data) or follow the same format for your custom dataset.
- `adaptive` flag enables running the adaptive AdaCon model, otherwise the static model is executed.
- `single` flag enables single-branch execution of AdaCon.
- `multi` flag running multi-branch execution of AdaCon (Default).
- `THRES` is the confidence threshold for AdaCon's branch controller (Default = 0.4).
- `IMG_SIZE` is the resolution of the input image.

### Training
train.py runs training for AdaCon model and it also supports training a static model. Our script takes a pretrained backbone (part of the static model), then it trains the branch controller as well as the branches.

```
python train.py --model {MODEL.args}
                 --data {DATA.data}
                 [--adaptive]
```
- `MODEL.args` is the path of the desired model description, check [model.args](https://github.com/scale-lab/AdaCon/blob/master/model.args) for an example.
- `DATA.data` is the path of the desired data, use [data/coco2014.data](https://github.com/scale-lab/AdaCon/blob/master/data/coco2014.data) and [data/coco2017.data](https://github.com/scale-lab/AdaCon/blob/master/data/coco2017.data) or follow the same format for your custom dataset.
- `adaptive` flag enables training the adaptive AdaCon model, otherwise the static model is trained.

## Citation

This code is a fork from [Ultralytics](https://github.com/ultralytics/yolov3) implementation for [Yolov3](https://pjreddie.com/darknet/yolo/).
