###  Train an Hybrid Task Cascade (HTC) with SWIN V2 using all datasets

# Load config file
from mmengine import Config
from mmengine.runner import set_random_seed
import torch
import subprocess
import os
from mmpretrain.models import SwinTransformerV2
import torch

from utils.functions_used_for_training import generate_unique_rgb, get_nb_classes_and_classes


# PARAMETERS
base_path = "Skinet/Projet/Data/Datasets/skinet_dataset"  # Dataset path 
nb_it = 1  # 1 iteration represents 10 epochs per dataset 
wanted_mode = 'inflammation'    # On which dataset we want to test our model at the end

prev_train = False
prev_train_path = ""

# Move wanted mode at the end of iterator
path_iterator = os.listdir(base_path)
del path_iterator[path_iterator.index(wanted_mode)]
path_iterator += [wanted_mode] 

# If you only want to train on 1 dataset
path_iterator =[wanted_mode] 

# Iterations
for step in range(nb_it):

    # Iterate over datasets
    for fold in path_iterator:

        # Current mode
        mode = fold
        mode_fold = os.path.join(base_path, fold)

        # Load config file
        cfg = Config.fromfile('Skinet/Projet/my_configs/used_configs/htc_swinv2.py')

        train_annotation_path = mode_fold + "/train/labels.json"

        # Get information about classes
        nb_classes, class_names = get_nb_classes_and_classes(train_annotation_path)
        palettes = generate_unique_rgb(nb_classes)

        # Change dataset classes and color in config file
        cfg.metainfo = {
            'classes': class_names,
            'palette': palettes
        }

        # Modify dataset type and path
        cfg.data_root = mode_fold

        # Informations about dataset in config file
        # Training
        cfg.train_dataloader.dataset.ann_file = 'train/labels.json'
        cfg.train_dataloader.dataset.data_root = cfg.data_root
        cfg.train_dataloader.dataset.data_prefix.img = 'train/data/'
        cfg.train_dataloader.dataset.metainfo = cfg.metainfo

        # Validation
        cfg.val_dataloader.dataset.ann_file = 'validation/labels.json'
        cfg.val_dataloader.dataset.data_root = cfg.data_root
        cfg.val_dataloader.dataset.data_prefix.img = 'validation/data/'
        cfg.val_dataloader.dataset.metainfo = cfg.metainfo

        # Test
        # Test dataloader can be the same as validation dataloader
        # If we dont have test data
        # cfg.test_dataloader = cfg.val_dataloader
        # Else
        cfg.test_dataloader.dataset.ann_file = 'test/labels.json'
        cfg.test_dataloader.dataset.data_root = cfg.data_root
        cfg.test_dataloader.dataset.data_prefix.img = 'test/data/'
        cfg.test_dataloader.dataset.metainfo = cfg.metainfo

        # Modify metric config
        cfg.val_evaluator.ann_file = cfg.data_root+'/'+'validation/labels.json'
        cfg.test_evaluator.ann_file = cfg.val_evaluator.ann_file
        cfg.test_evaluator.ann_file = cfg.data_root + '/' + 'test/labels.json'

        # Modify num classes of the model in box head and mask head
        for i in range(3):
            cfg.model.roi_head.bbox_head[i].num_classes = nb_classes
            cfg.model.roi_head.mask_head[i].num_classes = nb_classes

        # Load pretrained architecture
        # If start of training
        if not prev_train:
            cfg.load_from = None  # Can specify a pretained model found on web 
                                  # Or a pretrained model on skinet
            
        else: 
            # Start from a architecture checkpoint
            with open(prev_train_path+"/last_checkpoint", "r") as file:
                cfg.load_from = str(file.read())

        # Load a pretrained swin transformer
        # See list_pretrained_swin_v2.txt
        pretrained_swin = 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth'
        cfg.model.backbone.init_cfg = dict(checkpoint=pretrained_swin, type='Pretrained')
        cfg.pretrained = None

        # Swin parameters (from mmpretrain tool https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/backbones/swin_transformer_v2.py)
        # Custom import
        cfg.custom_imports = dict(allow_failed_imports=False, imports=['mmpretrain.models',])

        # Adapt depending on the version of swin you are using (see github link above )
        arch='tiny'
        window_size=8
        
        # Image size
        cfg.model.backbone.img_size=1024

        # Dont change anything here
        cfg.model.backbone.window_size=window_size
        cfg.model.backbone.type='mmpretrain.models.SwinTransformerV2'
        cfg.model.backbone.arch=arch
        cfg.model.backbone.pad_small_map=True #Always True for detection and segmentation   
        cfg.model.backbone.out_indices=(0,1,2,3)

        # Get output channels of each stage of the backbone Swin
        # With swin transformer from mmpretrain library
        in_channels_FPN = [] 

        # Parameters must be the same as above
        self = SwinTransformerV2(arch=arch, out_indices=(0,1,2,3), 
                            pad_small_map=True, window_size=window_size)
        
        # Dont change anything here
        self.eval()
        inputs = torch.rand(1, 3, 1500, 1500)
        output = self.forward(inputs)
        for level_out in output:
            in_channels_FPN.append(tuple(level_out.shape)[1])

        # Adapt neck 
        # the parameter in_channels in FPN must have same shape as swin transformer out_channels for each stage
        cfg.model.neck.in_channels = in_channels_FPN

        # Set up working dir to save files and logs for tensorboard
        cfg.work_dir = f'Skinet/Projet/exps/htc_swin_v2_all_data/step{step+1}/{mode}'
        prev_train_path = f'Skinet/Projet/exps/htc_swin_v2_all_data/step{step+1}/{mode}'

        # Workers per gpu
        cfg.train_dataloader.num_workers=2
        cfg.val_dataloader.num_workers=2
        cfg.test_dataloader.num_workers=2

        # Batchs
        cfg.train_dataloader.batch_size=2
        cfg.val_dataloader.batch_size=2
        cfg.test_dataloader.batch_size=2

        # Number of epochs per iteration and per dataset
        cfg.train_cfg.max_epochs=10

        # Set evaluation interval to reduce the evaluation times
        cfg.train_cfg.val_interval = 2

        # Set the checkpoint saving interval to reduce the storage cost
        cfg.default_hooks.checkpoint.interval = 10   # Number of epochs between checkpoints

        # Learning rate
        cfg.optim_wrapper.optimizer.lr = 0.0001

        # Iteration interval for printing log
        cfg.default_hooks.logger.interval = 10


        # Set seed thus the results are more reproducible
        # cfg.seed = 0
        set_random_seed(0, deterministic=False)

        # We can also use tensorboard to log the training process
        # Check if it is already present in the config file or not before uncomment
        # cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})

        # Write the configuration in a config file
        config='Skinet/Projet/my_configs/used_configs/htc_swinv2.py'
        with open(config, 'w') as f:
            f.write(cfg.pretty_text)


        # Command for training by opening a new terminal (Windows)
        # p = subprocess.Popen(["start", "cmd", "/k", "python Skinet/Projet/Code/mmdetection/tools/train.py " + config], shell=True)

        # Optimizing CUDA memory allocation
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" 

        # Command for training in IDE current terminal (Ubuntu) (Maybe it works on Windows too)
        p = os.system(f"python Skinet/Projet/Code/mmdetection/tools/train.py {config}")

        # For using checkpoint in next training
        prev_train = True



