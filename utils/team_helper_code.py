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
        gridsize = float(gridsize[0]) # returns a list, so take the first element
    else:
        raise Exception('No grid size available: have you saved the grid size in the header?')
    return gridsize

# Save the gridsize into header for a record. Assumes that gridsize is a number
def save_gridsize(record, gridsize):
    header_file = helper_code.get_header_file(record)
    header = helper_code.load_text(header_file)
    header += '# Gridsize: ' + str(gridsize) + '\n'
    helper_code.save_text(header_file, header)
    return header

# Save the rotation angle into header for a record. Assumes that rotation is a number
def save_rotation(record, gridsize):
    header_file = helper_code.get_header_file(record)
    header = helper_code.load_text(header_file)
    header += '# Rotation: ' + str(gridsize) + '\n'
    helper_code.save_text(header_file, header)
    return header

def find_available_images(ids : list, directory : str, verbose : bool):
    """
    Check for images or .npy arrays in the directory that match the IDs in the list.
    This compares the characters of the ID before '-' to avoid any additions in image, mask,
    or patch generation. (Somewhat of a hack, but works for the generator format.)

    params:
        ids: list of IDs to check for, e.g. ['01017_hr', '01018_hr', ...]
        directory: path to the directory to search
        verbose: printouts?

    returns:
        list of files that match requested ids, e.g. 
        ['01017_hr', '01018_hr', ...] or ['01017_hr-0', '01018_hr-0', ...]
    """
    ids = [f.split(os.sep)[-1] for f in ids] # Make sure IDs are strings and not paths
    ids = [f.split('-')[0] for f in ids] # take characters before '-0' 
    all_files = os.listdir(directory)
    image_ids = [f.split('.')[0] for f in all_files if (f.endswith('.png') or f.endswith('.npy'))]

    matching_image_ids = [f for f in image_ids if f.split('-')[0] in ids]
    if verbose and len(matching_image_ids) != len(ids):
        print(f"Some requested images are missing from {directory}. Using "+\
              f"{len(matching_image_ids)} images out of {len(ids)} requested.")

    return matching_image_ids

def check_dirs_for_ids(ids, dir1, dir2, verbose):
    """
    Check if all IDs in the list are present in one or both directories.
    This splits using the '-' character to avoid any additions (e.g. -0) in image, mask,
    or patch generation. (Somewhat of a hack, but works for the generator format.)
    """
    # Make sure IDs are strings and not paths
    ids = [f.split(os.sep)[-1] for f in ids]
    id_set = set([f.split('-')[0] for f in ids])

    dir1_files = os.listdir(dir1)
    dir1_set = set([f.split('-')[0] for f in dir1_files])
    dir2_set = set()
    if dir2:
        dir2_files = os.listdir(dir2)
        dir2_set = set([f.split('-')[0] for f in dir2_files])

    if dir2:
        available_pres = id_set.intersection(dir1_set, dir2_set)
    else:
        available_pres = id_set.intersection(dir1_set)
    available_ids = [f for f in ids if f.split('-')[0] in available_pres]

    if verbose and not id_set.issubset(dir1_set):
        if dir2 and not id_set.issubset(dir2_set):
            print(f"IDs requested do not match between directories. Using {len(available_pres)} " +\
                  f"out of {len(ids)} requested.")
        else:
            print(f"Not all IDs requested are present in {dir1}. Using {len(available_pres)} out " +\
                  f"of {len(ids)} requested.")
    return available_ids
    
