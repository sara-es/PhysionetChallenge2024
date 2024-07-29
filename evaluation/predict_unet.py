import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

import team_code, helper_code
import generator
from digitization import Unet
from utils import constants, team_helper_code, model_persistence
from evaluation import eval_utils
import pandas as pd


def generate_and_predict_unet_batch(images_folder, mask_folder, patch_folder,
                                  unet_output_folder, unet_model,
                                  verbose, records_to_process=None, delete_images=True,
                                  num_images_to_generate=0, data_folder=None):
    """
    An all-in-one to generate images from records, run them through the U-Net model, and 
    reconstruct the patches to a full image. Assumes we are generating these images, so we have 
    masks (labels), and can return a DICE score for evaluation.

    NOTE: used to be in team_code, but is currently not used - we train the resnet on the ground 
    truth data, not the U-net outputs.
    """
    if num_images_to_generate > 0:
        if verbose:
            print('Finding the Challenge data...')

        records = helper_code.find_records(data_folder)
        if len(records) == 0:
            raise FileNotFoundError('No data were provided.')
        # test on set number of records, set random_state for consistency if needed
        idx = 5000 # since I usually start from 0 for training with same seed
        records_to_process = shuffle(records, random_state=42)[idx:idx+num_images_to_generate] 

        # params for generating images
        img_gen_params = generator.DefaultArgs()
        img_gen_params.random_bw = 0.2
        img_gen_params.wrinkles = True
        img_gen_params.print_header = True
        img_gen_params.calibration_pulse = 1
        img_gen_params.augment = False
        img_gen_params.input_directory = wfdb_records_folder
        img_gen_params.output_directory = images_folder

        # set params for generating masks
        mask_gen_params = generator.MaskArgs()
        mask_gen_params.calibration_pulse = 1 # must be 0 or 1
        mask_gen_params.input_directory = data_folder
        mask_gen_params.output_directory = mask_folder

        # generate images 
        if verbose:
            print("Generating images from wfdb files...")
        generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process)
        if verbose:
            print("Generating masks from wfdb files...")
        generator.gen_ecg_images_from_data_batch.run(mask_gen_params, records_to_process)

        # generate patches
        Unet.patching.save_patches_batch(records_to_process, images_folder, mask_folder, constants.PATCH_SIZE, 
                                        patch_folder, verbose, max_samples=False)
    else:
        if not records_to_process:
            records_to_process = helper_code.find_records(images_folder)
        records_to_process = team_helper_code.check_dirs_for_ids(records_to_process, 
                                                                 images_folder, 
                                                                 mask_folder, verbose)

    dice_list, entropy_list = Unet.batch_predict_full_images(records_to_process, patch_folder, unet_model, 
                                   unet_output_folder, verbose, save_all=True)
    
    # save record name, dice list and entropy list to csv
    print(f"mean DICE: {np.mean(dice_list)}")
    print(f"dice list: {dice_list.shape}")
    print(f"entropy list: {entropy_list.shape}")
    df = pd.DataFrame({'record': records_to_process, 'dice': dice_list})
    df.to_csv(os.path.join(unet_output_folder, 'dice.csv'), index=False)

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


if __name__ == '__main__':
    # set up paths
    wfdb_records_folder = os.path.join("ptb-xl", "records500")
    images_folder = os.path.join('test_data', 'images')
    mask_folder = os.path.join('test_data', 'masks')
    patch_folder = os.path.join('test_data', 'patches')
    unet_output_folder = os.path.join('test_data', 'unet_outputs')

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(unet_output_folder, exist_ok=True)

    # unet_model = model_persistence.load_models("model", True, 
    #                     models_to_load=['digitization_model'])
    unet_model = model_persistence.load_checkpoint_dict("model", 
                                                        "UNET_256_checkpoint", True)

    num_images_to_generate = 0 # int, set to 0 if data has already been generated to speed up testing time

    # generate images from records, run through U-net, and reconstruct signals
    generate_and_predict_unet_batch(images_folder, mask_folder, patch_folder,
                                  unet_output_folder, unet_model,
                                  verbose=True, records_to_process=None, delete_images=False,
                                  num_images_to_generate=num_images_to_generate,
                                  data_folder=wfdb_records_folder)