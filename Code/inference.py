# Inference

### TODO ###  
# Recall is not perfect
# F1 score to compare with the previous solution (need to adapt IoU too)

import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import matplotlib.pyplot as plt
from PIL import Image
import os
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import torch
import sys
import io

from utils.functions_used_for_training import get_nb_classes_and_classes
from utils.functions_used_for_inference import *

# PARAMETERS
DISPLAY_RESULTS = True
COMPUTE_METRICS = True
# Choose to use a config and initialize the detector
config_file = 'Skinet/Projet/Data/trained_models/mask2former/mask2former_config.py'

# Setup a checkpoint file to load
checkpoint_file = 'Skinet/Projet/Data/trained_models/mask2former/trained_mask2former.pth'

# (Test) Data and its annotation paths
path = "Skinet/Projet/Data/Datasets/skinet_dataset/inflammation/test/data"
annotation_file = "Skinet/Projet/Data/Datasets/skinet_dataset/inflammation/test/labels.json"

# Register all modules in mmdet into the registries
register_all_modules() # Do not comment 

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Define the confidence score threshold
list_score_thresh = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
# SCORE_THRESHOLD = 0.  # For example 0.3

iou_threshold = 0.5  # IoU threshold to consider a prediction as a true positive

nb_classes, classes = get_nb_classes_and_classes(annotation_file)
print(classes)

# Initializing variables
predicted_labels = None
predicted_bboxes = None
predicted_masks = None
predicted_scores = None

# For saving metrics
sum_f1 = {class_name: 0.0 for class_name in classes}
sum_IoU = {class_name: 0.0 for class_name in classes}
sum_recall = {class_name: 0.0 for class_name in classes}

# For computing metrics based on the number of occurences of each class
nb_its = {class_name: 0 for class_name in classes}

it = 0


for SCORE_THRESHOLD in  [0.3] :
    for it,img in enumerate(os.listdir(path)):

        if "AJI2983B1_02" in img:

            # print(f"Iteration {it} on {len(os.listdir(path))}")
            img_path = os.path.join(path, img)

            # Use the detector to do inference
            image = mmcv.imread(img_path, channel_order='rgb')

            # Inference
            result = inference_detector(model, image)
            
            # Get inference's results
            predicted_labels = result.pred_instances.labels.cpu()
            predicted_bboxes = result.pred_instances.bboxes.cpu()
            predicted_masks = result.pred_instances.masks.cpu()
            predicted_scores = result.pred_instances.scores.cpu()

            # Load the JSON annotations file
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            coco = COCO(annotation_file)
            sys.stdout = old_stdout

            # Find the image ID based on its file name
            imgIds = coco.getImgIds()
            imgs = coco.loadImgs(imgIds)
            image_id = None
            for image in imgs:
                if image['file_name'] == img:
                    image_id = image['id']
                    img_info = image  # Save the image info
                    break

            if image_id is None:
                raise ValueError("Image ID not found for the given file name.")

            # Load the image
            original_image = Image.open(img_path).convert('RGB')

            print(predicted_labels.unique())
            unique_labels = torch.cat((predicted_labels, torch.tensor([ann['category_id'] - 1  for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))]))).unique()
            
            # For each detected class
            for unique_label in unique_labels:

                class_name = classes[unique_label]
                
                if DISPLAY_RESULTS :
                    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                    for ax in axs:
                        ax.axis('off')

                # Filter predictions by class and score threshold
                class_indices = (predicted_labels == unique_label) & (predicted_scores > SCORE_THRESHOLD)
                class_predicted_masks = predicted_masks[class_indices]
                class_predicted_bboxes = predicted_bboxes[class_indices]
                class_predicted_scores = predicted_scores[class_indices]

                # Display for the predicted class
                if DISPLAY_RESULTS :
                    if len(class_predicted_masks) > 0:
                        predicted_image_with_masks = apply_masks_and_draw_contours(original_image, class_predicted_masks)
                        axs[0].imshow(predicted_image_with_masks)
                        axs[0].set_title(f"Prediction Class {class_name}")
                    else:
                        axs[0].set_title(f"No predicted instances for {class_name} or confidence score is too low")

                # Get image annotations for this class
                ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[unique_label+1])
                annotations = coco.loadAnns(ann_ids)

                # Get ground truth masks for the current class
                ground_truth_masks = []
                for ann in annotations:
                    if 'segmentation' in ann:
                        rle = coco_mask.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                        mask = coco_mask.decode(rle)
                        if mask.ndim == 3:
                            mask = mask[..., 0]
                        ground_truth_masks.append(mask)

                # Apply ground truth masks and display
                if DISPLAY_RESULTS:
                    if ground_truth_masks:
                        ground_truth_image_with_masks = apply_masks_and_draw_contours(original_image, ground_truth_masks, draw_contours=True)
                        axs[1].imshow(ground_truth_image_with_masks)
                    else:
                        axs[1].imshow(original_image)  # Display the original image if there are no masks

                    axs[1].set_title(f"Ground Truth Class {class_name}")

                if COMPUTE_METRICS :

                    # Calculate true positives, false positives, and false negatives
                    true_positives = 0
                    false_positives = len(class_predicted_masks)  # Initially, all predicted masks are considered as false positives
                    false_negatives = len(ground_truth_masks)    # Initially, all ground truth masks are considered as false negatives
                    iou_sum = 0  # Sum of IoUs for true positive matches
                    
                    # For each instance of the current class
                    for pred_mask in class_predicted_masks:

                        # Iterate through each predicted mask
                        for gt_mask in ground_truth_masks:

                            # Compare the predicted mask with each ground truth mask
                            iou = calculate_iou(pred_mask, gt_mask)

                            # If the Intersection over Union (IoU) is greater than the threshold,
                            iou_sum += iou  # Add IoU to the sum for this mask

                            # It is considered a match (true positive)
                            if iou > iou_threshold:
                                true_positives += 1
                                false_positives -= 1   # One less false positive as we found a match
                                false_negatives -= 1  # One less false negative as we found a match
                                break  # Assume that one predicted mask corresponds to only one ground truth mask

                    # Calculate the average IoU for the class (if there are true positives)
                    average_iou = iou_sum / true_positives if true_positives > 0 else 0

                    # Calculate and print the F1 score for the class
                    TP, TP_plus_FN, f1_score = calculate_f1_score(true_positives, false_positives, false_negatives)

                    # Calculate the total number of instances for the class in ground truths
                    total_instances = len(ground_truth_masks)

                    # print(f"Classe {class_name}: Score F1 = {f1_score:.2f}, Average IoU = {average_iou:.2f}")
                    # print(f"Correctly Detected Instances for Class {class_name}: {TP}/{TP_plus_FN}")

                    # Sums for metrics' mean
                    sum_f1[class_name] += f1_score
                    sum_IoU[class_name] += average_iou
                    sum_recall[class_name] += TP/TP_plus_FN if TP_plus_FN > 0 else 0
                    nb_its[class_name] += 1 if TP_plus_FN > 0 else 0

                if DISPLAY_RESULTS :
                    plt.show()

    # Display metrics
    if COMPUTE_METRICS :
        print(f"SCORE_THRESHOLD : {SCORE_THRESHOLD} ")
        print(f"IoU_THRESHOLD : {iou_threshold}\n ")

        print("NB Iterations :\n ")
        print(nb_its)
        print("F1 SCORE :")
        print({key : round(value / nb_its[key], 2) for key,value in sum_f1.items() if nb_its[key] != 0 }, "\n")
        print("IoUs : ")
        print({key : round(value / nb_its[key], 2) for key,value in sum_IoU.items() if nb_its[key] != 0}, "\n"  )
        print("RECALL : ")
        print({key : round(value / nb_its[key], 2) for key,value in sum_recall.items() if nb_its[key] != 0}, "\n"  )