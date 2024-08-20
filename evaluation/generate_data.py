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

    records_to_process = helper_code.find_records(data_folder)
    if max_samples is not None:
        records_to_process = shuffle(records_to_process, random_state=42)[20:max_samples]
    num_records = len(records_to_process)

    # if num_records == 0:
    #     raise FileNotFoundError('No data were provided.')

    images_folder = os.path.join(output_folder, "images")
    masks_folder = os.path.join(output_folder, "masks")
    patch_folder = os.path.join(output_folder, "patches")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(patch_folder, exist_ok=True)

    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.input_directory = data_folder
    img_gen_params.output_directory = images_folder
    img_gen_params.random_bw = 0.2
    img_gen_params.wrinkles = True
    img_gen_params.print_header = True
    img_gen_params.augment = True
    img_gen_params.crop = 0.0
    img_gen_params.rotate = 0
    img_gen_params.lead_bbox = True
    img_gen_params.lead_name_bbox = True
    img_gen_params.store_config = 2

    # img_gen_params.deterministic_noise = True
    # img_gen_params.noise=0

    # img_gen_params.augment = False
    img_gen_params.calibration_pulse = 0

    # set params for generating masks
    # mask_gen_params = generator.MaskArgs()
    # mask_gen_params.input_directory = wfdb_records_folder
    # mask_gen_params.output_directory = masks_folder
    # mask_gen_params.calibration_pulse = 0

    # generate images - params done manually because the generator doesn't implement seed correctly
    split = int(len(records_to_process)/4) # 25% no calibration pulse, 25% no noise/wrinkles, 50% with rotation
    records_to_process = shuffle(records_to_process)
    if verbose:
        print("Generating images from wfdb files (set 1/3)...")
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process[:split])
    img_gen_params.calibration_pulse = 1
    if verbose:
        print("Generating images from wfdb files (set 2/4)...")
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process[split:int(split*2)])
    img_gen_params.rotate = 10
    if verbose:
        print("Generating images from wfdb files (set 3/4)...")
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process[int(split*2):int(split*3)])
    img_gen_params.wrinkles = False
    img_gen_params.augment = False
    if verbose:
        print("Generating images from wfdb files (set 4/4)...")    
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process[int(split*3):])

    # generate masks
    # if verbose:
    #     print("Generating masks from wfdb files (set 1/2)...")
    # generator.gen_ecg_images_from_data_batch.run(mask_gen_params, records_to_process[:split])
    # mask_gen_params.calibration_pulse = 1
    # if verbose:
    #     print("Generating masks from wfdb files (set 2/2)...")
    # generator.gen_ecg_images_from_data_batch.run(mask_gen_params, records_to_process[split:])
    
    # # generate patches
    # Unet.patching.save_patches_batch(records, images_folder, masks_folder, constants.PATCH_SIZE, 
    #                                  patch_folder, verbose, delete_images=False, max_samples=40000)

    if verbose:
        print("Done.")


if __name__ == "__main__":
    data_folder = os.path.join("ptb-xl", "records500")
    # data_folder = "G:\\PhysionetChallenge2024\\tiny_testset\\lr_gt"
    # data_folder = os.path.join("temp_data", "train", "images")
    output_folder_prefix = os.path.join("test_data") # will create images, masks, patches subfolders here
    verbose = True
    max_samples = 40 # set to None to train on all available

    # training data: no rotation on images, also generates json with config.
    # patches images and masks, but does not run unet
    generate_training_data(data_folder, output_folder_prefix, verbose, max_samples)

