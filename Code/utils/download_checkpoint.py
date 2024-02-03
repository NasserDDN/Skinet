# Get informations about a model and its pretrained files
import os
from mim import get_model_info

infos = get_model_info('mmdet', models=['mask r-cnn'],
               shown_fields=['epoch', 'training_memory', 'coco/box_ap', 'coco/mask_ap'],
               filter_conditions='epochs>30')

print(infos)

# Download a pretrained file
p = os.system(f"cd Data && mim download mmdet --config mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco --dest ../Data/checkpoints ")


import subprocess
# Windows command
# p = subprocess.Popen(["start", "cmd", "/k", "cd Data && mim download mmdet --config mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco --dest ../Data/checkpoints"], shell = True)
