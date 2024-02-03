import random
import json


# Generate rgb palette (1 color per class)
def generate_unique_rgb(num: int):
    """
    Generate rgb palette (1 color per class)

    PARAMETERS
        num : Number of classes

    RETURN
        generated (tuple) : Tuple of colors
    """
    generated = []
    while len(generated) < num:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        generated.append((r, g, b))
    return generated

def get_nb_classes_and_classes(path: str):
    """
    Get number of classes and a tuple of classes from a json annotation file

    PARAMETERS
        path : Path to json annotation file

    RETURN
        unique_classes_nb (int) : Number of classes
        class_names (tuple) : Tuple of classes
    """

    with open(path, 'r') as file:
        data = json.load(file)

    categories = data['categories']
    class_names = tuple(category['name'] for category in categories)

    unique_classes_nb = len(categories)

    return unique_classes_nb, class_names