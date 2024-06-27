"""
Generates all data locally for testing purposes: images, masks, patches of images and masks,
unet output images, and reconstructed signals.

By default this uses signals from tiny_testset, but this can be changed by setting the data_folder
argument.
"""

import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import matplotlib.pyplot as plt
import numpy as np
import team_code, helper_code
from sklearn.utils import shuffle
import generator


def generate_data(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    num_records = len(records)
    records = shuffle(records, random_state=42)[:5] # test on a tiny number of records for now

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    images_folder = os.path.join("test_data", "images")
    masks_folder = os.path.join("test_data", "masks")
    patch_folder = os.path.join("test_data", "patches")
    unet_output_folder = os.path.join("test_data", "unet_outputs")
    reconstructed_signals_folder = os.path.join("test_data", "reconstructed_signals")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(unet_output_folder, exist_ok=True)
    os.makedirs(reconstructed_signals_folder, exist_ok=True)

    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.random_bw = 0.2
    img_gen_params.wrinkles = True
    img_gen_params.print_header = True
    img_gen_params.calibration_pulse = 0.5
    img_gen_params.lead_bbox = True
    img_gen_params.lead_name_bbox = True
    img_gen_params.store_config = 1
    img_gen_params.input_directory = data_folder
    img_gen_params.output_directory = images_folder

    # generate images and masks
    if verbose:
        print("Generating images from wfdb files...")
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records)



if __name__ == "__main__":
    data_folder = os.path.join("ptb-xl", "records500")
    # data_folder = "G:\\PhysionetChallenge2024\\tiny_testset\\lr_gt"
    # data_folder = os.path.join("test_data", "images")
    model_folder = os.path.join("model")
    verbose = True

    generate_data(data_folder, model_folder, verbose)