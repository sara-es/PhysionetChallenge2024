"""
Assumes that images (unet outputs) have already been generated and saved in the unet_output_folder.
"""
import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy as sp
import skimage
from utils import team_helper_code


def test_unet_rotation(record_ids, unet_outputs_dir, images_dir, verbose=True):
    # find the available record IDs
    record_ids = team_helper_code.check_dirs_for_ids(record_ids, images_dir, unet_outputs_dir, verbose)
    unet_output_ids = team_helper_code.find_available_images(record_ids, unet_outputs_dir, verbose)

    pred_rotations = []
    true_rotations = []

    min_angle = -8
    max_angle = 8

    for i, record in enumerate(unet_output_ids):
        unet_output_path = os.path.join(unet_outputs_dir, record + ".npy")
        with open(unet_output_path, 'rb') as f:
            test_im = np.load(f)

        mask_path = os.path.join("test_rot_data", "masks", record + ".npy")
        with open(unet_output_path, 'rb') as f:
            mask = np.load(f)

        active = np.shape(test_im)[1] # set to image width
        rot_angle = 1000

        # closing filter on the image to remove noise
        test_im = skimage.morphology.closing(test_im, footprint=[(np.ones((5, 1)), 1), (np.ones((1, 5)), 1)])

        # plt.imshow(test_im)
        # plt.show()

        for angle in range(min_angle, max_angle): # for debugging, this is only searching -5 to +4 degrees
            rot_image = sp.ndimage.rotate(test_im, angle, axes=(1, 0), reshape=True)
            # rot_image[rot_image < 0] = 0 # binarize
            col_hist = np.sum(rot_image, axis = 0) #sum each column 

            # find the starting and end column - columns with black pixels within the active region
            idxs = np.where(col_hist == 0)[0]
            if len(idxs) == 0:
                col_hist = np.round(col_hist, decimals=12) # if col sum has no black lines, try within a tolerance
                idxs = np.where(col_hist == 0)[0]
                if len(idxs) == 0: # if it STILL fails, just continue
                    continue
            startcol = idxs[0]
            endcol = idxs[-1]
            this_active = endcol-startcol
            
            if this_active < active:
                active = this_active
                rot_angle = angle

        # load rotation angle info from json
        config_file = os.path.join(images_dir, record + "-0.json")
        with open(config_file, 'r') as f:
            config = json.load(f)
        true_rotation = config['rotate']
        true_rotations.append(true_rotation)

        print(f"Record {record} true rotation: {true_rotation}, predicted rotation: {rot_angle}")

if __name__ == '__main__':
    record_ids = [f.split(".")[0] for f in os.listdir("test_rot_data\\unet_outputs")]
    unet_outputs_dir = "test_rot_data\\unet_outputs"
    images_dir = "test_rot_data\\images"
    test_unet_rotation(record_ids, unet_outputs_dir, images_dir, verbose=True)