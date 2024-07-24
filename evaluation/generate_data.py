"""
Generates all data locally for testing purposes: images, masks, patches of images and masks,
unet output images, and reconstructed signals.

By default this uses signals from tiny_testset, but this can be changed by setting the data_folder
argument.
"""

import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import helper_code
from sklearn.utils import shuffle
from digitization import Unet
import generator
from utils import constants


def generate_training_data(data_folder, output_folder, verbose, max_samples):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    if max_samples is not None:
        records = shuffle(records, random_state=42)[:max_samples]
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    images_folder = os.path.join(output_folder, "images")
    masks_folder = os.path.join(output_folder, "masks")
    patch_folder = os.path.join(output_folder, "patches")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(patch_folder, exist_ok=True)

    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.random_bw = 0.2
    img_gen_params.wrinkles = True
    img_gen_params.print_header = True
    img_gen_params.augment = True
    # img_gen_params.rotate = 8
    # even with seed, pulse is not deterministic, must be 0 or 1 to match masks
    img_gen_params.calibration_pulse = 1
    img_gen_params.store_config = 2
    img_gen_params.seed = 42
    img_gen_params.input_directory = data_folder
    img_gen_params.output_directory = images_folder

    # set params for generating masks
    mask_gen_params = generator.MaskArgs()
    mask_gen_params.seed = 42
    mask_gen_params.calibration_pulse = 1 # must be 0 or 1
    mask_gen_params.input_directory = data_folder
    mask_gen_params.output_directory = masks_folder

    # generate images and masks
    if verbose:
        print("Generating images from wfdb files...")
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records)
    if verbose:
        print("Generating masks from wfdb files...")
    generator.gen_ecg_images_from_data_batch.run(mask_gen_params, records)
    
    # generate patches
    Unet.patching.save_patches_batch(records, images_folder, masks_folder, constants.PATCH_SIZE, 
                                     patch_folder, verbose, max_samples=False)

    if verbose:
        print("Done.")


if __name__ == "__main__":
    data_folder = os.path.join("ptb-xl", "records500")
    # data_folder = "G:\\PhysionetChallenge2024\\tiny_testset\\lr_gt"
    # data_folder = os.path.join("test_data", "images")
    output_folder_prefix = "temp_data" # will create images, masks, patches subfolders here
    verbose = True
    max_samples = 200 # set to None to train on all available

    # training data: no rotation on images, also generates json with config.
    # patches images and masks, but does not run unet
    generate_training_data(data_folder, output_folder_prefix, verbose, max_samples)

