# SKINET (Segmentation of the Kidney threw a Neuronal Network)

## Installation

### Environment
First, clone this repo.

Create a new conda environment with the .yml file : 

```
conda create --name openmmlab --file environment.yml
```

Go to Code folder and enter the following commands (click [here](https://mmdetection.readthedocs.io/en/latest/get_started.html) for more details):
```
pip install -U openmim
mim install "mmengine==0.9.0"
mim install "mmcv==2.0.1"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

### Previous Skinet environment
This project is using the previous Skinet project to generate datasets.

Use it to generate the 5 datasets correspondings to the 5 different modes.

Two different folders are generated for each dataset (training and validation data). Create a third folder for the test part and move some images from the validation folder into this third folder.

Then use the **Code/utils/generate_annotation_coco_format.py** script to generate annotations from the datasets and to copy input images in another directories. You have to adapt the 6 folder paths at the beginning of the script for each datasets. Adapt the paths to obtain a structure identical to the **Data/Datasets/skinet_datasets** folder:

```
previous_skinet (previous skinet's solution)
├── ...
└── ...
Skinet
  └── Projet/
    ├── Code/
    ├── mmdetection/
    │ ├── utils/
    │ └── ...python scripts
    ├── Data/
    │ ├── checkpoints/
    │ ├── Dataset/
    | | └── skinet_dataset/
    | |   ├── inflammation
    | |   | ├── test
    | |   | | ├── data
    | |   | | └── labels.json
    | |   | ├── train
    | |   | | ├── data
    | |   | | └── labels.json
    | |   | └── validation
    | |   |   ├── data
    | |   |   └── labels.json
    | |   └── ...other modes/
    │ └── trained_models/
    |   ├── htc
    |   └── mask2former
    └── my_configs/
      └── used_configs/
```

## Inference
You can download some pretrained models to make inferences. (### Put a link here ###).
Then you can adapt the inference script by changing paths and use it.
Please refer to the inference_tutorial file for more information. 

## Train
There are three training scripts: one for training an HTC with a Swin Transformer v1 as the backbone, another for training an HTC with a Swin v2 as the backbone, and a third for training a Mask2Former with a Swin v1 as the backbone.
You can use these scripts by adapting the Parameters (comments explain the utility of each parameter).
If you wish to modify the configuration files, please refer to the mmdetection_tutorial file for explanations.
