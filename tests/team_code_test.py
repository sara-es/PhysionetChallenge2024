import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import joblib, time
from tqdm import tqdm
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet

def train_models(data_folder, model_folder, verbose):
    """
    Team code version
    """
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    # Get the file paths of signals
    # tts = 0.6
    # records = shuffle(records)
    # train_records = records[:int(tts*num_records)]
    # val_records = records[int(tts*num_records):]

    # use tiny testset for testing
    data_folder = os.path.join("tiny_testset", "lr_unet_tests", "data_images")
    records = helper_code.find_records(data_folder)
    num_records = len(records)
    train_records = records[:10]
    val_records = records[10:]
    images_folder = os.path.join("tiny_testset", "lr_unet_tests", "data_images")
    masks_folder = os.path.join("tiny_testset", "lr_unet_tests", "binary_masks")
    image_patch_folder = os.path.join("tiny_testset", "lr_unet_tests", "image_patches")
    mask_patch_folder = os.path.join("tiny_testset", "lr_unet_tests", "label_patches") 

    print(train_records)

    # generate images and masks for training u-net; generate patches
    # images_folder = os.path.join("ptb-xl", "train_images")
    # masks_folder = os.path.join("ptb-xl", "train_masks")
    
    # team_code.generate_unet_training_data(data_folder, images_folder, 
    #                                       masks_folder, patches_folder, 
    #                                       train_records, verbose)

    # train u-net
    args = Unet.utils.Args()
    args.train_val_prop = 0.8
    args.epochs = 2
    team_code.train_unet(train_records, image_patch_folder, mask_patch_folder, model_folder, verbose, 
                         args=args, warm_start=True)

    # save trained u-net

    # generate new images

    # run u-net on new images

    # reconstruct signals from u-net outputs

    # train digitization model

    # save trained classification model

    # optionally display some results

    # optionally delete generated images, masks, and patches




data_folder = "G:\\PhysionetChallenge2024\\ptb-xl\\combined_records"
model_folder = "G:\\PhysionetChallenge2024\\model"
verbose = True

if __name__ == "__main__":
    train_models(data_folder, model_folder, verbose)