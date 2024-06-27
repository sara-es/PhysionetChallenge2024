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
from sklearn.utils import shuffle
from digitization import Unet
import generator
from utils import constants, team_helper_code, model_persistence
from tqdm import tqdm


def generate_data(data_folder, model_folder, verbose):

    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    num_records = len(records)
    records = shuffle(records, random_state=42)[:50] # test on a tiny number of records for now

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    images_folder = os.path.join("test_rot_data", "images")
    masks_folder = os.path.join("test_rot_data", "masks")
    patch_folder = os.path.join("test_rot_data", "patches")
    unet_output_folder = os.path.join("test_rot_data", "unet_outputs")
    # reconstructed_signals_folder = os.path.join("test_rot_data", "reconstructed_signals")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(unet_output_folder, exist_ok=True)
    # os.makedirs(reconstructed_signals_folder, exist_ok=True)

    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.random_bw = 0.2
    img_gen_params.wrinkles = True
    img_gen_params.print_header = True
    img_gen_params.augment = True
    img_gen_params.rotate = 8
    img_gen_params.calibration_pulse = 0.5
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
    
    # load u-net
    models = model_persistence.load_models(model_folder, verbose, models_to_load=['digitization_model'])
    unet_state_dict = models['digitization_model']
    # checkpoint = torch.load("model\\UNET_256_checkpoint")
    # unet_state_dict = checkpoint['model_state_dict']
    dice_list = Unet.batch_predict_full_images(records, patch_folder, unet_state_dict, 
                                   unet_output_folder, verbose, save_all=True)
    print(np.asarray(dice_list).mean())

    # reconstruct_signals
    if verbose:
        print("Reconstructing signals from u-net outputs...")
    reconstructed_signals = []
    for record in tqdm(records):
        # load u-net outputs
        record_id = team_helper_code.find_available_images(
                            [record], unet_output_folder, verbose)[0] # returns list
        unet_image_path = os.path.join(unet_output_folder, record_id + '.npy')
        with open(unet_image_path, 'rb') as f:
            unet_image = np.load(f)

        # save unet output as image to check (can comment this out if not needed)
        with open(unet_image_path.replace('.npy', '.png'), 'wb') as f:
            plt.imsave(f, unet_image, cmap='gray')

        # reconstruct signal
        # load header file to save with reconstructed signal
        # record_path = os.path.join(data_folder, record) 
        # header_txt = helper_code.load_header(record_path)
        # # print(helper_code.get_baselines(header_txt))
        # # print(helper_code.get_adc_zeros(header_txt))
        # rec_signal, _ = team_code.reconstruct_signal(record_id, unet_image, header_txt, 
        #                reconstructed_signals_folder)
        # reconstructed_signals.append(rec_signal)  

    if verbose:
        print("Done.")


if __name__ == "__main__":
    data_folder = "G:\\PhysionetChallenge2024\\ptb-xl\\records500"
    # data_folder = "G:\\PhysionetChallenge2024\\tiny_testset\\lr_gt"
    # data_folder = os.path.join("test_data", "images")
    model_folder = "G:\\PhysionetChallenge2024\\model"
    verbose = True

    generate_data(data_folder, model_folder, verbose)
