###  Train a mask2Former architecture using all datasets

### TODO ###
# The pretrained backbone should not be loaded because a pretrained mask2former is already loaded
# Try training without loading a pretrained Swin Transformer

# Load config file
from mmengine import Config
from mmengine.runner import set_random_seed
import subprocess
import torch
import os

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
path_iterator = [wanted_mode] 


# Iterations
for step in range(nb_it):

    # Iterate over datasets
    for fold in path_iterator:

        # Current mode
        mode = fold
        mode_fold = os.path.join(base_path, fold)

        # Load config file
        cfg = Config.fromfile('Skinet/Projet/my_configs/used_configs/mask2former_swinv1.py')

        train_annotation_path = mode_fold + "/train/labels.json"

        # Get information about classes
        nb_classes, class_names = get_nb_classes_and_classes(train_annotation_path)
        palettes = generate_unique_rgb(nb_classes)

        # Change dataset classes and color in config file
        cfg.metainfo = {
            'classes': class_names,
            'palette': palettes
        }

        # Modify dataset path
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

        # Test dataloader can be the same as validation dataloader
        # If we dont have test data
        # cfg.test_dataloader = cfg.val_dataloader
        
        cfg.test_dataloader.dataset.ann_file = 'test/labels.json'
        cfg.test_dataloader.dataset.data_root = cfg.data_root
        cfg.test_dataloader.dataset.data_prefix.img = 'test/data/'
        cfg.test_dataloader.dataset.metainfo = cfg.metainfo

        # Modify metric config
        cfg.val_evaluator.ann_file = cfg.data_root+'/'+'validation/labels.json'
        cfg.test_evaluator.ann_file = cfg.val_evaluator.ann_file
        cfg.test_evaluator.ann_file = cfg.data_root + '/' + 'test/labels.json'
        
        # Load pretrained architecture
        # If start of training
        if not prev_train:
            cfg.load_from = "Skinet/Projet/Data/checkpoints/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"
            
        else:
            # Start from a architecture checkpoint
            with open(prev_train_path+"/last_checkpoint", "r") as file:
                cfg.load_from = str(file.read())

        # Load a pretrained swin transformer
        pretrained_swin = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
        cfg.model.backbone.init_cfg = dict(checkpoint=pretrained_swin, type='Pretrained')
        
        # Modify num classes in the config file
        cfg.num_things_classes = nb_classes
        cfg.num_stuff_classes = 0
        cfg.num_classes = nb_classes + cfg.num_stuff_classes

        # In panoptic fusion head
        cfg.model.panoptic_fusion_head.num_stuff_classes = 0
        cfg.model.panoptic_fusion_head.num_things_classes = nb_classes

        # In panoptic head
        cfg.model.panoptic_head.num_stuff_classes = 0
        cfg.model.panoptic_head.num_things_classes = nb_classes

        # Weights of classes
        # Addition of a low weight representing the background
        cfg.model.panoptic_head.loss_cls.class_weight =[1.0] * nb_classes + [0.1]  

        # Set up working dir to save files and logs for tensorboard
        cfg.work_dir = f'Skinet/Projet/exps/mask2former_swin_s_alldata/step{step+1}/{mode}'
        prev_train_path = f'Skinet/Projet/exps/mask2former_swin_s_alldata/step{step+1}/{mode}'
        
        # Workers per gpu
        cfg.train_dataloader.num_workers=2
        cfg.val_dataloader.num_workers=2
        cfg.test_dataloader.num_workers=2

        # Batchs
        cfg.train_dataloader.batch_size=1
        cfg.val_dataloader.batch_size=1
        cfg.test_dataloader.batch_size=1

        # Number of epochs per iteration and per dataset
        cfg.train_cfg.max_epochs=1

        # Never change
        cfg.default_scope = 'mmdet'

        # Image size
        img_size = (1024, 1024)
        cfg.image_size = img_size

        # Set evaluation interval to reduce the evaluation times
        cfg.train_cfg.val_interval = 1

        # Set the checkpoint saving interval to reduce the storage cost
        cfg.default_hooks.checkpoint.interval = 10   # Number of epochs between checkpoints 

        # Learning rate
        cfg.optim_wrapper.optimizer.lr = 0.0001

        # Iteration interval for printing log in terminal
        cfg.default_hooks.logger.interval = 10

        # Set seed thus the results are more reproducible
        # cfg.seed = 0
        set_random_seed(0, deterministic=False)

        # We can also use tensorboard to log the training process
        # Check if it is already present in the config file or not before uncomment
        # cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})

        # Write the configuration in a config file
        config='Skinet/Projet/my_configs/used_configs/mask2former_swinv1.py'
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



