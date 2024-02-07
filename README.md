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
