import os
import json
import cv2
import shutil

# Parameters
copy_input_images_to_another_dir = False
generate_annotation = True


def calculate_annotation_polygon(mask_path):
    # Load the mask image in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour represents the object
    contour = max(contours, key=cv2.contourArea)

    # Flatten the contour array and convert it to a list
    segmentation = contour.flatten().tolist()

    # Calculate bounding box
    x, y, w, h = cv2.boundingRect(contour)
    bbox = [x, y, w, h]
    
    # Calculate area
    area = cv2.contourArea(contour)

    return segmentation, area, bbox

def generate_coco_json(base_path,train,train_categories):
    # Structure to hold COCO data
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Helper variables
    image_id = 1
    annotation_id = 1
    category_id = 1
    category_mapping = {}

    # Iterating over each image folder
    for image_folder in os.listdir(base_path):
        image_folder_path = os.path.join(base_path, image_folder)

        # Check if it's a directory
        if not os.path.isdir(image_folder_path):
            continue

        # Path to the original image
        original_image_path = os.path.join(image_folder_path, "full_images")

        # Assuming there's only one image in the 'full_images' folder
        for file_name in os.listdir(original_image_path):
            if file_name.lower().endswith('.jpg'):
                # Add image info to COCO data
                file_path = os.path.join(original_image_path, file_name)
                image_info = {
                    "id": image_id,
                    "file_name": file_name,
                    "width": 1024,
                    "height": 1024
                }
                coco_data["images"].append(image_info)
                break

        # Process each class folder in the image folder
        for class_folder in os.listdir(image_folder_path):
            class_folder_path = os.path.join(image_folder_path, class_folder)

            # Ignore the 'full_images' and 'images' folders
            if class_folder in ["full_images", "images", "cortex"] or not os.path.isdir(class_folder_path):
                continue

            # If training annotations (To be able to have same categories list for each annotations file)
            if train:
                # Assign a category ID if new
                if class_folder not in category_mapping:
                    category_mapping[class_folder] = category_id
                    coco_data["categories"].append({
                        "id": category_id,
                        "name": class_folder
                    })
                    category_id += 1
            else:
                # Validation annotation has the sames categories as train annotation 
                coco_data["categories"] = train_categories
                for cat in train_categories:
                    category_mapping[cat['name']] = cat['id']  

            # Process each mask in the class folder
            for mask_file in os.listdir(class_folder_path):
                if mask_file.lower().endswith('.jpg'):
                    mask_path = os.path.join(class_folder_path, mask_file)

                    # Calculate polygon segmentation, area, bbox for each mask
                    segmentation, area, bbox = calculate_annotation_polygon(mask_path)

                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_mapping[class_folder],
                        "segmentation": [segmentation],  # Polygon format
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation_info)
                    annotation_id += 1

        image_id += 1

    return coco_data

if generate_annotation:
    # Generate coco format annotation json
    base_path = "previous_skinet/nephrology_inflammation_dataset_train"
    train_annotation = generate_coco_json(base_path,train=True, train_categories="")
    annnotation_path = 'Skinet/Projet/Data/Datasets/skinet_dataset/inflammation/train_with_cortex/labels.json'
    default_data ={} 
    if not os.path.exists(annnotation_path):
        os.makedirs(os.path.dirname(annnotation_path), exist_ok=True)

        with open(annnotation_path, 'x') as file:
            json.dump(default_data, file, indent=4)
    with open( annnotation_path, 'w') as file:
        json.dump(train_annotation, file, indent=4)

    print(train_annotation['categories'])
    print("train annotations generated")

    base_path = "previous_skinet/nephrology_inflammation_dataset_val"
    val_annotation = generate_coco_json(base_path, train=False, train_categories=train_annotation['categories'] )
    annnotation_path = 'Skinet/Projet/Data/Datasets/skinet_dataset/inflammation/validation_with_cortex/labels.json'

    if not os.path.exists(annnotation_path):
        os.makedirs(os.path.dirname(annnotation_path), exist_ok=True)

        with open(annnotation_path, 'x') as file:
            json.dump(default_data, file, indent=4)

    with open( annnotation_path, 'w') as file:
        json.dump(val_annotation, file, indent=4)

    print("validation annotations generated")

    base_path = "previous_skinet/nasser_files_test_images_1024"
    val_annotation = generate_coco_json(base_path, train=False, train_categories=train_annotation['categories'] )
    annnotation_path = 'Skinet/Projet/Data/Datasets/skinet_dataset/inflammation/test_with_cortex/labels.json'

    if not os.path.exists(annnotation_path):
        os.makedirs(os.path.dirname(annnotation_path), exist_ok=True)

        with open(annnotation_path, 'x') as file:
            json.dump(default_data, file, indent=4)

    with open( annnotation_path, 'w') as file:
        json.dump(val_annotation, file, indent=4)

    print("test annotations generated")


# Copy input images ('full_images' subfolder in images folder) in an another directory
def copy_full_images(source_directory, target_directory):
    # Create target folder if none exists
    os.makedirs(target_directory, exist_ok=True)

    # Browse source directory
    for root, dirs, files in os.walk(source_directory):
        if 'full_images' in root:
            for file in files:
                # Check if the file is an image
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Full path to source file
                    source_file_path = os.path.join(root, file)

                    # Full path to target file
                    target_file_path = os.path.join(target_directory, file)

                    # Copy file
                    shutil.copy2(source_file_path, target_file_path)
                    print(f"Copié: {source_file_path} -> {target_file_path}")

# Example of use
if copy_input_images_to_another_dir:
    source_directory = 'previous_skinet/nephrology_mest_glom_dataset_train'
    target_directory = 'Skinet/Projet/Data/Datasets/skinet_dataset/mest_glom/train/data'
    copy_full_images(source_directory, target_directory)

    source_directory = 'previous_skinet/nephrology_mest_glom_dataset_val'
    target_directory = 'Skinet/Projet/Data/Datasets/skinet_dataset/mest_glom/validation/data'
    copy_full_images(source_directory, target_directory)

    source_directory = 'previous_skinet/test_mest_glom'
    target_directory = 'Skinet/Projet/Data/Datasets/skinet_dataset/mest_glom/test/data'
    copy_full_images(source_directory, target_directory)
