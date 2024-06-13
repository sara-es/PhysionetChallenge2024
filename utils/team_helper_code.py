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

# Get the gridsize from a header or a similar string.
def get_gridsize_from_header(string):
    gridsize, has_gridsize = helper_code.get_variables(string, '# Gridsize:')
    if has_gridsize:
        gridsize = float(gridsize)
    else:
        raise Exception('No labels available: are you trying to load the labels from the held-out data, or did you forget to prepare the data to include the labels?')
    return gridsize

# Save the gridsize into header for a record. Assumes that gridsize is a number
def save_gridsize(record, gridsize):
    header_file = helper_code.get_header_file(record)
    header = helper_code.load_text(header_file)
    header += '#Gridsize: ' + str(gridsize) + '\n'
    helper_code.save_text(header_file, header)
    return header
