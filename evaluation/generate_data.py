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
import torch
import team_code, helper_code
from evaluation import eval_utils
from sklearn.utils import shuffle
from digitization import Unet
import generator
from utils import constants, team_helper_code, model_persistence
from tqdm import tqdm


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
    # img_gen_params.calibration_pulse = 1
    img_gen_params.store_config = 2
    img_gen_params.input_directory = data_folder
    img_gen_params.output_directory = images_folder

    # set params for generating masks
    mask_gen_params = generator.MaskArgs()
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


def generate_and_predict_unet_batch(wfdb_records_folder, images_folder, mask_folder, patch_folder,
                                  unet_output_folder, unet_model, reconstructed_signals_folder,
                                  verbose, records_to_process=None, delete_images=True):
    """
    An all-in-one to generate images from records, run them through the U-Net model, and 
    reconstruct the patches to a full image. Assumes we are generating these images, so we have 
    masks (labels), and can return a DICE score for evaluation.

    NOTE: used to be in team_code, but is currently not used - we train the resnet on the ground 
    truth data, not the U-net outputs.
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(wfdb_records_folder)

    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.random_bw = 0.2
    img_gen_params.wrinkles = True
    img_gen_params.print_header = True
    img_gen_params.input_directory = wfdb_records_folder
    img_gen_params.output_directory = images_folder

    # generate images 
    if verbose:
        print("Generating images from wfdb files...")
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process)

    # generate patches
    Unet.patching.save_patches_batch(records_to_process, images_folder, mask_folder, constants.PATCH_SIZE, 
                                     patch_folder, verbose, max_samples=False)
    dice_list = Unet.batch_predict_full_images(records_to_process, patch_folder, unet_model, 
                                   unet_output_folder, verbose, save_all=True)

    # reconstruct_signals
    reconstructed_signals = []
    snr_list = np.zeros(len(records_to_process))
    for i, record in tqdm(enumerate(records_to_process), 
                       desc='Reconstructing signals from U-net outputs', disable=not verbose):
        # load u-net outputs
        record_id = team_helper_code.find_available_images(
                            [record], unet_output_folder, verbose)[0] # returns list
        unet_image_path = os.path.join(unet_output_folder, record_id + '.npy')
        with open(unet_image_path, 'rb') as f:
            unet_image = np.load(f)

        # reconstruct signal
        # load header file to save with reconstructed signal
        record_path = os.path.join(wfdb_records_folder, record) 
        label_signal, label_fields = helper_code.load_signals(record_path)
        header_txt = helper_code.load_header(record_path)
        rec_signal, _ = team_code.reconstruct_signal(record_id, unet_image, header_txt, 
                       reconstructed_signals_folder)
        reconstructed_signals.append(rec_signal)  

        snr_list[i], _, _, _, _ = eval_utils.single_signal_snr(rec_signal, label_fields, 
                                    label_signal, label_fields, record, extra_scores=False)

    if verbose:
        print(f"Average DICE score: {np.mean(dice_list)}")
        print(f"Average SNR: {np.mean(snr_list)}")

    # optional: delete training images and masks, patches, and u-net outputs
    if delete_images:
        im_patch_dir = os.path.join(patch_folder, 'image_patches')
        label_patch_dir = os.path.join(patch_folder, 'label_patches')
        for im in os.listdir(im_patch_dir):
            os.remove(os.path.join(im_patch_dir, im))
        for im in os.listdir(label_patch_dir):
            os.remove(os.path.join(label_patch_dir, im))
        for im in os.listdir(images_folder):
            os.remove(os.path.join(images_folder, im))
        for im in os.listdir(mask_folder):
            os.remove(os.path.join(mask_folder, im))
        for im in os.listdir(unet_output_folder):
            os.remove(os.path.join(unet_output_folder, im))


if __name__ == "__main__":
    data_folder = os.path.join("ptb-xl", "records500")
    # data_folder = "G:\\PhysionetChallenge2024\\tiny_testset\\lr_gt"
    # data_folder = os.path.join("test_data", "images")
    output_folder_prefix = "temp_data" # will create images, masks, patches subfolders here
    verbose = True
    max_samples = 50 # set to None to train on all available

    # training data: no rotation on images, also generates json with config.
    # patches images and masks, but does not run unet
    generate_training_data(data_folder, output_folder_prefix, verbose, max_samples)

