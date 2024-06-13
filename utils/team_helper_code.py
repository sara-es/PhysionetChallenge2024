import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np

import helper_code

def load_image_paths(record):
    path = os.path.split(record)[0]
    image_files = helper_code.get_image_files(record)

    images = list()
    for image_file in image_files:
        image_file_path = os.path.join(path, image_file)
        if os.path.isfile(image_file_path):
            images.append(image_file_path)

    return images