#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most 
# parts of the required functions, change or remove non-required functions, and add your own 
# functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib, os, sys, time
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.utils import shuffle

import helper_code
import preprocessing
from utils import team_helper_code, constants, model_persistence
from digitization import Unet, ECGminer
from classification import seresnet18
import generator, preprocessing, digitization, classification
import generator.gen_ecg_images_from_data_batch
from evaluation import eval_utils

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of 
# the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code,
# but do *not* change the arguments of this function. If you do not train one of the models, then
# you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    start_time = time.time()
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Train the digitization model. If you are not training a digitization model, then you can
    # remove this part of the code.
    if verbose:
        print('Training the digitization model...')

    digitization_model = train_digitization_model(data_folder, model_folder, verbose, 
                                records_to_process=records, delete_training_data=False)
    
    if verbose:
        time1 = time.time()
        print(f'Done. Time to train digitization model: ' + \
              f'{time1 - start_time:.2f} seconds.')
    
    # Extract the features and labels from the data.
    if verbose:
        print('Training the classification model...')

    classification_model, classes = train_classification_model(data_folder, verbose, 
                                                               records_to_process=None)

    if verbose:
        time2 = time.time()
        print(f'Done. Time to train classification model: ' + \
              f'{time2 - time1:.2f} seconds.')
        
    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_models(model_folder, digitization_model, classification_model, classes)

    if verbose:
        print('Done. Total time to train models: ' + f'{time.time() - start_time:.2f} seconds.')
        print()


# Load your trained models. This function is *required*. You should edit this function to add your
# code, but do *not* change the arguments of this function. If you do not train one of the models,
# then you can return None for the model.
def load_models(model_folder, verbose):
    models = model_persistence.load_models(model_folder, verbose, 
                        models_to_load=['digitization_model', 'classification_model', 'dx_classes'])
    digitization_model = models['digitization_model']
    # classification_model = models['classification_model', 'dx_classes']
    return digitization_model, models


# Run your trained digitization model. This function is *required*. You should edit this function
# to add your code, but do *not* change the arguments of this function. If you did not train one of
# the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Load the digitization model.
    unet_model = Unet.utils.load_unet_from_state_dict(digitization_model)

    # Preprocess the image to get rotation angle
    # preprocess_with_unet_predict(record, unet_model, verbose)

    # Run the digitization model; if you did not train this model, then you can set signal=None.
    signal, reconstructed_signal_dir = unet_reconstruct_single_image(record, unet_model, verbose, 
                                                                 delete_patches=True)
    
    # Load the classification model and classes.
    resnet_model = classification_model['classification_model']
    dx_classes = classification_model['dx_classes']
    
    # Run the classification model; if you did not train this model, then you can set labels=None.
    labels = classify_signals(record, reconstructed_signal_dir, resnet_model, 
                                dx_classes, verbose=verbose)
    
    # delete any temporary files
    for f in os.listdir(reconstructed_signal_dir):
        os.remove(os.path.join(reconstructed_signal_dir, f))
    
    return signal, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    if digitization_model is not None:
        model_persistence.save_model_torch(digitization_model, 'digitization_model', model_folder)

    if classification_model is not None:
        d = {'classification_model': classification_model, 'dx_classes': classes}
        model_persistence.save_models(d, model_folder, verbose=False)
        

def train_digitization_model(data_folder, model_folder, verbose, records_to_process=None,
                             delete_training_data=True, max_size_training_set=2000):
    """
    Our general digitization process is
    1. generate testing images and masks
    2. preprocess testing images to estimate grid size/scale
    3. generate u-net patches
    4. run u-net on patches
    5. recover full image with signal outline from u-net outputs
    6. reconstruct signal and trace from u-net output

    At each step we save the outputs to disk to save on memory; here, we assume by default that 
    they should be deleted when no longer needed, but if you want to keep them for debugging or
    visualization, set delete_training_data to False. 
    """
    # hard code some folder paths for now
    images_folder = os.path.join(os.getcwd(), 'temp_data', 'images')
    masks_folder = os.path.join(os.getcwd(), 'temp_data', 'masks')
    patch_folder = os.path.join(os.getcwd(), 'temp_data', 'patches')
    unet_output_folder = os.path.join(os.getcwd(), 'temp_data', 'unet_outputs')

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(unet_output_folder, exist_ok=True)

    # TODO can do a split here if we want to have unet train and predict on different records
    if not records_to_process:
        records_to_process = helper_code.find_records(data_folder)
    if max_size_training_set is not None:
        records_to_process = shuffle(records_to_process)[:max_size_training_set]

    # generate images and masks for training u-net; generate patches
    generate_unet_training_data(data_folder, images_folder, 
                                masks_folder, patch_folder, 
                                verbose, records_to_process=records_to_process)
    if verbose:
        print(f'Done.')
    
    # train U-net
    args = Unet.utils.Args()
    args.train_val_prop = 1.0 # we want to train on all available data
    args.epochs = 50 # SET THIS IN FINAL SUBMISSION
    checkpoint_folder = os.path.join('digitization', 'model_checkpoints')
    unet_model = train_unet(records_to_process, patch_folder, checkpoint_folder, verbose, args=args, 
                            warm_start=True)
    if verbose:
        print(f'Done.')
    
    return unet_model
        

def generate_unet_training_data(wfdb_records_folder, images_folder, masks_folder, patch_folder,
                                verbose, patch_size=constants.PATCH_SIZE, records_to_process=None):
    """
    Call generate_images_from_wfdb to generate images and masks; then patchify the images and masks
    for training the U-Net model. Save the patches in patches_folder. 
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(wfdb_records_folder)

    seed = np.random.randint(100000) # DOES NOTHING APPARENTLY >:C

    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.seed = seed
    img_gen_params.random_bw = 0.2
    img_gen_params.wrinkles = True
    img_gen_params.print_header = True
    img_gen_params.augment = True
    img_gen_params.calibration_pulse = 1
    img_gen_params.input_directory = wfdb_records_folder
    img_gen_params.output_directory = images_folder

    # set params for generating masks
    mask_gen_params = generator.MaskArgs()
    mask_gen_params.seed = seed
    mask_gen_params.calibration_pulse = 1
    mask_gen_params.input_directory = wfdb_records_folder
    mask_gen_params.output_directory = masks_folder

    # generate images and masks WITH and WITHOUT reference pulse
    split = int(len(records_to_process)/2)
    records_to_process = shuffle(records_to_process)
    if verbose:
        print("Generating images from wfdb files...")
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process[:split])
    img_gen_params.calibration_pulse = 0
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process[split:])
    if verbose:
        print("Generating masks from wfdb files...")
    generator.gen_ecg_images_from_data_batch.run(mask_gen_params, records_to_process[:split])
    mask_gen_params.calibration_pulse = 0
    generator.gen_ecg_images_from_data_batch.run(mask_gen_params, records_to_process[split:])

    # generate patches
    Unet.patching.save_patches_batch(records_to_process, images_folder, masks_folder, patch_size,
                                     patch_folder, verbose, max_samples=False)


def train_unet(record_ids, patch_folder, model_folder, verbose, 
               args=None, max_train_samples=5000, warm_start=True, delete_patches=True):
    """
    Train the U-Net model from patches and save the resulting model. 
    Note that no validation is done by default - during the challenge we will want to train
    on all available data. Manually set args.train_val_prop to a value between 0 and 1 to
    enforce validation.
    """
    if not args: # use default args if none are provided
        args = Unet.utils.Args()
    
    patchsize = constants.PATCH_SIZE
    # path for model checkpoints, used with early stopping or to resume training later
    CHK_PATH_UNET = os.path.join(model_folder, 'UNET_' + str(patchsize))
    # for saving the loss values, used with early stopping
    LOSS_PATH = os.path.join(model_folder, 'UNET_' + str(patchsize) + '_losses')
    # if we're loading a pretrained model - hardcoded for now
    LOAD_PATH_UNET = None
    if warm_start:
        chkpt_path = os.path.join('digitization', 'model_checkpoints', 
                                      'UNET_'+ str(patchsize) + '_checkpoint')
        if not os.path.exists(chkpt_path):
            print(f"Warm start requested but no checkpoint found at {LOAD_PATH_UNET}, " +\
                  "training U-net from scratch.")
        else:
            LOAD_PATH_UNET = chkpt_path

    image_patch_folder = os.path.join(patch_folder, 'image_patches')
    mask_patch_folder = os.path.join(patch_folder, 'label_patches')

    unet_model = Unet.train_unet(record_ids, image_patch_folder, mask_patch_folder,
            args, CHK_PATH_UNET, LOSS_PATH, LOAD_PATH_UNET, verbose,
            max_samples=max_train_samples,
            )
    
    if delete_patches: 
        for im in os.listdir(image_patch_folder):
            os.remove(os.path.join(image_patch_folder, im))
        for im in os.listdir(mask_patch_folder):
            os.remove(os.path.join(mask_patch_folder, im))

    return unet_model


def reconstruct_signal(record, unet_image, header_txt, 
                       reconstructed_signals_folder, save_signal=True):
    """
    reconstruct signals from u-net outputs

    Returns:
        reconstructed_signal: pandas dataframe, reconstructed signal
        raw_signals: np.array, raw signals in pixel coords
        gridsize: float, scaling factor for the signals in pixel units
    """
    signal_length = helper_code.get_num_samples(header_txt)
    fs = helper_code.get_sampling_frequency(header_txt)
    max_duration = int(signal_length/fs)
    # max duration on images cannot exceed 10s as per Challenge team
    max_duration = 10 if max_duration > 10 else max_duration 
    reconstructed_signal, raw_signals, gridsize  = ECGminer.digitize_image_unet(unet_image, 
                                    sig_len=signal_length, max_duration=max_duration)
    reconstructed_signal = np.asarray(np.nan_to_num(reconstructed_signal))

    # save reconstructed signal and copied header file in the same folder
    if save_signal:
        output_record_path = os.path.join(reconstructed_signals_folder, record)
        helper_code.save_header(output_record_path, header_txt)
        comments = [l for l in header_txt.split('\n') if l.startswith('#')]
        helper_code.save_signals(output_record_path, reconstructed_signal, comments)

    # TODO: adapt self.postprocessor.postprocess to work for different layouts
    # return raw_signals and gridsize for external evaluation
    return reconstructed_signal, raw_signals, gridsize


def train_classification_model(records_folder, verbose, records_to_process=None):
    """
    Extracts features and labels from headers, then one-hot encodes labels and trains the
    SE-ResNet model.
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(records_folder)

    all_data = []
    labels = []
    for record in tqdm(records_to_process, desc='Loading classifier training data', 
                       disable=not verbose):
        # TODO need to make sure that data is interpolated/downsampled to consistent frequency
        data, label = classification.get_training_data(record, records_folder)
        if label is None: # don't use data without labels for training
            continue

        all_data.append(data)
        labels.append(label)
    
    # Make sure we actually have data and labels
    if len(all_data) == 0:
        raise ValueError("No records with labels found in records to process.")

    # One-hot-encode labels 
    mlb = MultiLabelBinarizer()
    multilabels = mlb.fit_transform(labels)
    uniq_labels = mlb.classes_

    if verbose:
        print("Training SE-ResNet classification model...")
    resnet_model = seresnet18.train_model(
                                all_data, multilabels, uniq_labels, verbose, epochs=5, 
                                validate=False)
    
    if verbose:
        print("Finished training classification model.")
    
    return resnet_model, uniq_labels


def unet_reconstruct_single_image(record, model, verbose, delete_patches=True):
    """
    params
        record: str, relative path from data folder and record ID, 
            e.g. 'ptbl-xl/records500/01017_hr'
        model: U-net state dict
        verbose: bool
        delete_patches: bool, whether to delete patches after processing    
    """
    # get image from image_path
    image = helper_code.load_images(record)[0]
    record_id = os.path.split(record)[-1].split('.')[0]

    # hard code some folder paths for now
    patch_folder = os.path.join('temp_data', 'patches', 'test_image_patches')
    reconstructed_signals_folder = os.path.join('temp_data', 'reconstructed_signals')
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(reconstructed_signals_folder, exist_ok=True)

    # patchify image
    image = np.asarray(image) # convert to numpy array
    Unet.patching.save_patches_single_image(record_id, image, None, 
                                            patch_size=constants.PATCH_SIZE,
                                            im_patch_save_path=patch_folder,
                                            lab_patch_save_path=None)

    # predict on patches, recover u-net output image
    predicted_image = Unet.predict_single_image(record_id, patch_folder, model,
                                                original_image_size=image.shape[:2])
    
    # rotate reconstructed u-net output to original orientation
    predicted_image, rot_angle = preprocessing.column_rotation(record_id, predicted_image,
                                                    angle_range=(-45, 45), verbose=verbose)
    
    if rot_angle != 0: # re-patch and predict on the rotated image (TODO: check if necessary)
        image = sp.ndimage.rotate(image, rot_angle, axes=(1, 0), reshape=True)
        Unet.patching.save_patches_single_image(record_id, image, None, 
                                            patch_size=constants.PATCH_SIZE,
                                            im_patch_save_path=patch_folder,
                                            lab_patch_save_path=None)

        # predict on patches, recover u-net output image
        predicted_image = Unet.predict_single_image(record_id, patch_folder, model,
                                                original_image_size=image.shape[:2])

    # reconstruct signal from u-net output image
    # load header file to save with reconstructed signal
    header_txt = helper_code.load_header(record)
    reconstructed_signal, raw_signals, _ = reconstruct_signal(record_id, predicted_image, 
                                                     header_txt,
                                                     reconstructed_signals_folder, 
                                                     save_signal=True)
    # if reconstructed_signal is None and trace is None:


    # optional: delete patches
    if delete_patches:
        for im in os.listdir(patch_folder):
            os.remove(os.path.join(patch_folder, im))

    # return reconstructed signal
    return reconstructed_signal, reconstructed_signals_folder


def classify_signals(record_path, data_folder, resnet_model, classes, verbose):
    # wrap in list to match training data format
    record_id = os.path.split(record_path)[-1].split('.')[0]
    data = [classification.get_testing_data(record_id, data_folder)] 
    pred_dx, probabilities = seresnet18.predict_proba(
                                        resnet_model, data, classes, verbose)
    labels = classes[np.where(pred_dx == 1)]
    if verbose:
        print(f"Classes: {classes}, probabilities: {probabilities}")
        print(f"Predicted labels: {labels}")

    return labels

    

