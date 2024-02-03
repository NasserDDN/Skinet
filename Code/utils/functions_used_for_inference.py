import numpy as np
import torch
import cv2
from PIL import Image


def apply_masks_and_draw_contours(image, masks, draw_contours=True):

    image_array = np.array(image)
    combined_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)

    # Combine instances masks
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        combined_mask = np.maximum(combined_mask, mask.astype(np.uint8) * 255)

    # Draw contours for each instances
    if draw_contours:
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_array, contours, -1, (0, 255, 0), 2)

    # Apply color on each mask
    red_mask = np.zeros_like(image_array, dtype=np.uint8)
    red_mask[combined_mask > 0] = [0, 0, 255]

    # Interpolate input image and masks predictions
    blended_image = Image.blend(image, Image.fromarray(red_mask), alpha=0.5)
    result = Image.blend(Image.fromarray(image_array), blended_image, alpha=1.)

    return result

def calculate_iou(mask1, mask2):

    # Check if the masks are PyTorch tensors; if yes, convert them to NumPy arrays
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()  # Convert to NumPy array
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()  # Convert to NumPy array

    # Calculate intersection and union of the two masks
    intersection = np.logical_and(mask1, mask2)  # Intersection: pixels that are True in both masks
    union = np.logical_or(mask1, mask2)         # Union: pixels that are True in either mask

    # Compute the IoU (Intersection over Union)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_f1_score(true_positives, false_positives, false_negatives):

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    # Calculate the F1 score using the precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    TP_p_FN = true_positives + false_negatives

    return true_positives, TP_p_FN, f1
