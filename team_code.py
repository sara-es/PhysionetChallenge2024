#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you 
# can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib, os, sys, time
import numpy as np
from tqdm import tqdm
import preprocessing
import helper_code
from utils import team_helper_code, constants
from digitization import Unet, ECGminer

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the 
# arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to 
# add your code, but do *not* change the arguments of this function. If you do not 
# train one of the models, then you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print('Training the digitization model...')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    digitization_features = list()
    classification_features = list()
    classification_labels = list()

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image; this simple example uses the same features for the digitization and classification
        # tasks.
        features = extract_features(record)
        
        digitization_features.append(features)

        # Some images may not be labeled...
        labels = helper_code.load_labels(record)
        if labels:
            classification_features.append(features)
            classification_labels.append(labels)

    # ... but we expect some images to be labeled for classification.
    if not classification_labels:
        raise Exception('There are no labels for the data.')

    # Train the models.
    if verbose:
        print('Training the models on the data...')

    # Train the digitization model. This very simple model uses the mean of these very simple features as a seed for a random number
    # generator.
    digitization_model = np.mean(features)

    # Train the classification model. This very simple model trains a random forest model with these very simple features.

    classification_features = np.vstack(classification_features)
    classes = sorted(set.union(*map(set, classification_labels)))
    classification_labels = helper_code.compute_one_hot_encoding(classification_labels, classes)

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state   = 56  # Random state; set for reproducibility.

    # Fit the model.
    # classification_model = RandomForestClassifier(
    #     n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(classification_features, classification_labels)

    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    # save_models(model_folder, digitization_model, classification_model, classes)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):
    digitization_model = dict()
    # digitization_filename = os.path.join(model_folder, 'digitization_model.sav')
    # TODO: make this robust to different saved unet models
    # TODO: load patch size from file
    digitization_model['unet'] = Unet.utils.load_unet_from_state_dict(model_folder, None)

    classification_model = dict()
    # classification_filename = os.path.join(model_folder, 'classification_model.sav')
    # classification_model = joblib.load(classification_filename)
    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Run the digitization model; if you did not train this model, then you can set signal = None.
    model = digitization_model['model']

    # Extract features.
    features = extract_features(record)

    # Load the dimensions of the signal.
    header_file = helper_code.get_header_file(record)
    header = helper_code.load_text(header_file)

    num_samples = helper_code.get_num_samples(header)
    num_signals = helper_code.get_num_signals(header)

    # Generate "random" waveforms using the a random seed from the feature.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1, high=1, size=(num_samples, num_signals))
    
    # Run the classification model.
    model = classification_model['model']
    classes = classification_model['classes']

    # Extract features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # Get model probabilities.
    probabilities = model.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    # Choose the class or classes with the highest probability as the label or labels.
    max_probability = np.nanmax(probabilities)
    labels = [classes[i] for i, probability in enumerate(probabilities) if probability == max_probability]

    return signal, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    images = helper_code.load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    if digitization_model is not None:
        d = {'model': digitization_model}
        filename = os.path.join(model_folder, 'digitization_model.sav')
        joblib.dump(d, filename, protocol=0)

    if classification_model is not None:
        d = {'model': classification_model, 'classes': classes}
        filename = os.path.join(model_folder, 'classification_model.sav')
        joblib.dump(d, filename, protocol=0)


def generate_images_from_wfdb(records_folder, images_folder, generation_params, verbose, 
                              records_to_process=None):
    """TODO
    Use WFDB records found in records_folder to generate images and save them in images_folder.
    Optionally provide a list of a subset of records to process (records_to_process).
    """
    pass


def preprocess_images(raw_images_folder, processed_images_folder, verbose, 
                      records_to_process=None):
    """
    Preprocess images found in raw_images_folder and save them in processed_images_folder.
    Optionally provide a list of a subset of records to process (records_to_process).

    Preprocessing steps currently implemented:
        - resize images to a standard size
    
    Todo:
        - determine grid size of the image and save it to either the original header file 
        (will need to pass in the header file path, wfdb_records_folder) or a new file
    """
    if not records_to_process:
        records_to_process = os.listdir(raw_images_folder)
    
    for i in tqdm(range(len(records_to_process)), desc='Preprocessing images', disable=~verbose):
        record = records_to_process[i] #FIXME this will break - need to get the record id from the record path
        raw_image_path = os.path.join(raw_images_folder, record + '.png')
        processed_image = os.path.join(processed_images_folder, record + '.png')
        # load raw image
        images = helper_code.load_images(raw_image_path)
        images = preprocessing.resize_images(images)
        


def generate_unet_training_data(wfdb_records_folder, images_folder, masks_folder, patches_folder,
                                verbose, patch_size=constants.PATCH_SIZE, records_to_process=None,
                                delete_images=False):
    """TODO
    Call generate_images_from_wfdb to generate images and masks; then patchify the images and masks
    for training the U-Net model. Save the patches in patches_folder. Option to delete images and 
    masks (not the patches) after patchifying to save space.
    """
    pass


def train_unet(record_ids, patch_folder, model_folder, verbose, 
               args=None, max_train_samples=5000, warm_start=False, delete_patches=True):
    """
    Train the U-Net model from patches and save the resulting model. 
    Note that no validation is done by default - during the challenge we will want to train
    on all available data. Manually set args.train_val_prop to a value between 0 and 1 to
    enforce validation.
    """
    if not args: # use default args if none are provided
        args = Unet.utils.Args()
    
    patchsize = constants.PATCH_SIZE
    # where the model will be saved
    PATH_UNET = os.path.join(model_folder, 'UNET_' + str(patchsize))
    # path for model checkpoints, used with early stopping
    CHK_PATH_UNET = os.path.join(model_folder, 'UNET_' + str(patchsize) + '_checkpoint')
    # for saving the loss values, used with early stopping
    LOSS_PATH = os.path.join(model_folder, 'UNET_' + str(patchsize) + '_losses')
    # if we're loading a pretrained model - hardcoded for now
    LOAD_PATH_UNET = None
    if warm_start:
        chkpt_path = os.path.join('model', 'pretrained', 
                                      'UNET_run1_'+ str(patchsize) + '_checkpoint')
        if not os.path.exists(chkpt_path):
            print(f"Warm start requested but no model found at {LOAD_PATH_UNET}, " +\
                  "training U-net from scratch.")
        else:
            LOAD_PATH_UNET = chkpt_path

    image_patch_folder = os.path.join(patch_folder, 'image_patches')
    mask_patch_folder = os.path.join(patch_folder, 'label_patches')

    Unet.train_unet(record_ids, image_patch_folder, mask_patch_folder, args,
            PATH_UNET, CHK_PATH_UNET, LOSS_PATH, LOAD_PATH_UNET, verbose,
            max_samples=max_train_samples,
            )
    
    if delete_patches:
        for im in os.listdir(image_patch_folder):
            os.remove(os.path.join(image_patch_folder, im))
        for im in os.listdir(mask_patch_folder):
            os.remove(os.path.join(mask_patch_folder, im))


def unet_predict_single_image(record_id, image, patch_folder, model, reconstructed_signal_folder,
                              verbose, delete_patches=True):
    """
    params
        record_id: str, for saving
        
    """
    # get image from image_path


    # preprocess image

    # patchify image
    # TODO will need to save/load patch size for persistence

    image_patches_array, _ = Unet.patching.save_patches_single_image(
                                        record_id, image, None, 
                                        patch_size=(constants.PATCH_SIZE, constants.PATCH_SIZE),
                                        im_patch_save_path=patch_folder,
                                        label_patch_save_path=None
                                        )

    # predict on patches
    predicted_image = Unet.predict_single_image(record_id, patch_folder, model)

    # reconstruct signal from patches

    # optional: save reconstructed signal

    # optional: delete patches

    # return reconstructed signal


def reconstruct_signal(unet_image, gridsize):
    """Input - an image (mask) from the unet, where ECG = 1, background - 0
       Output - 
       
       There are three stages:
           1. convert unet image into a set of Point (ecg-miner) objects that correspond to each row of ECG, in pixels
           2. identify and remove any reference pulses
           3. convert into individual ECG channels and convert from pixels to mV
    """
    #TODO: adapt self.postprocessor.postprocess to work for different layouts
    ECG_miner.digitize_image.digitize_image_unet(unet_image, gridsize, sig_len=1000)


def generate_resnet_training_data(wfdb_records_folder, images_folder, mask_folder, patch_folder,
                                  unet_output_folder, model_folder, reconstructed_signals_folder, 
                                  verbose, records_to_process=None, delete_images=True):
    """
    An all-in-one to generate images from records, run them through the U-Net model, and 
    reconstruct the signal. Assumes we are generating these images, so we have masks (labels).

    TODO: can either move a lot of these folder names to constants, or hard code them from a base 
    directory since they're all temporary files anyway
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(wfdb_records_folder)

    # TODO: fill out generation params and uncomment below
    test_generation_params = {}
    # generate_images_from_wfdb(wfdb_records_folder, images_folder, test_generation_params, 
    #                           verbose, records_to_process)
    # preprocess_images(images_folder, images_folder, verbose, records_to_process)
    
    # TODO: write load_model and move this code there
    # Load the model
    model = Unet.utils.load_unet_from_state_dict(model_folder)

    # generate patches
    Unet.patching.save_patches_batch(images_folder, mask_folder, constants.PATCH_SIZE, 
                                     patch_folder, verbose, max_samples=False)
    Unet.batch_predict_full_images(records_to_process, patch_folder, model_folder, 
                                   unet_output_folder, verbose, save_all=True)

    # reconstruct_signals
    # save reconstructed signals
    # delete image patches
    # optional: delete images and patches



def train_classifier():
    pass

def classify_signal():
    pass

