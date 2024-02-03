# Remove cortex class from a JSON annotation file

import json

# Path to original JSON file
input_json_path = 'Skinet/Projet/Data/Datasets/skinet_dataset/mest_main/validation/labels.json'

# Path of modified JSON file
output_json_path = 'Skinet/Projet/Data/Datasets/skinet_dataset/mest_main/validation/labels.json'

def renumber_classes(json_file, output_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Create a new ID mapping for categories
    new_id_map = {}
    new_categories = []
    new_id = 1
    for category in data['categories']:
        new_id_map[category['id']] = new_id
        category['id'] = new_id
        new_categories.append(category)
        new_id += 1

    # Update annotations with new category IDs
    for annotation in data['annotations']:
        annotation['category_id'] = new_id_map[annotation['category_id']]

    # Update the list of categories in the data
    data['categories'] = new_categories

    # Save the modified JSON file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

# Load JSON file
with open(input_json_path, 'r') as file:
    data = json.load(file)

# Find 'cortex' category ID
cortex_id = None
for category in data['categories']:
    if category['name'] == 'cortex':
        cortex_id = category['id']
        break

# Check if the category 'cortex' has been found
if cortex_id is not None:
    # Filter annotations, keeping only those not in the 'cortex' category
    data['annotations'] = [anno for anno in data['annotations'] if anno['category_id'] != cortex_id]

    # Remove category 'cortex' from category list
    data['categories'] = [cat for cat in data['categories'] if cat['id'] != cortex_id]

    # Save the modified JSON file
    with open(output_json_path, 'w') as file:
        json.dump(data, file, indent=4)
else:
    print("Category 'cortex' not found.")

# Arrangement of class ids because a class has been deleted
renumber_classes(input_json_path, output_json_path)
