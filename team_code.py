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
import preprocessing
import helper_code
from utils import team_helper_code, constants, model_persistence
from preprocessing.resize_images import resize_images
from digitization import Unet, ECGminer
from classification import seresnet18
import classification

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
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Train the digitization model. If you are not training a digitization model, then you can
    # remove this part of the code.

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

        # Extract the features from the image; this simple example uses the same features for the
        # digitization and classification tasks.
        features = extract_features(record)
        
        digitization_features.append(features)

        # Some images may not be labeled...
        labels = load_labels(record)
        if any(label for label in labels):
            classification_features.append(features)
            classification_labels.append(labels)

    # ... but we expect some images to be labeled for classification.
    if not classification_labels:
        raise Exception('There are no labels for the data.')

    # Train the models.
    if verbose:
        print('Training the models on the data...')

    # Train the digitization model. This very simple model uses the mean of these very 
    # simple features as a seed for a random number generator.
    digitization_model = np.mean(features)

    # Train the classification model. If you are not training a classification model, then you can
    # remove this part of the code.
    
    # This very simple model trains a random forest model with these very simple features.
    classification_features = np.vstack(classification_features)
    classes = sorted(set.union(*map(set, classification_labels)))
    classification_labels = compute_one_hot_encoding(classification_labels, classes)

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state   = 56  # Random state; set for reproducibility.

    # Fit the model.
    classification_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(classification_features, classification_labels)

    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_models(model_folder, digitization_model, classification_model, classes)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your
# code, but do *not* change the arguments of this function. If you do not train one of the models,
# then you can return None for the model.
def load_models(model_folder, verbose):
    digitization_filename = os.path.join(model_folder, 'digitization_model.sav')
    digitization_model = joblib.load(digitization_filename)

    classification_filename = os.path.join(model_folder, 'classification_model.sav')
    classification_model = joblib.load(classification_filename)
    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function
# to add your code, but do *not* change the arguments of this function. If you did not train one of
# the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Run the digitization model; if you did not train this model, then you can set signal=None.

    # Load the digitization model.
    model = digitization_model['model']

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # Generate "random" waveforms using the a random seed from the features.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1, high=1, size=(num_samples, num_signals))
    
    # Run the classification model; if you did not train this model, then you can set labels=None.

    # Load the classification model and classes.
    model = classification_model['model']
    classes = classification_model['classes']

    # Get the model probabilities.
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
    
    TODO:
        - determine grid size of the image and save it to either the original header file 
        (will need to pass in the header file path, wfdb_records_folder) or a new file
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(raw_images_folder)
    
    for i in tqdm(range(len(records_to_process)), desc='Preprocessing images', disable=~verbose):
        record = records_to_process[i] #FIXME this will break - need to get the record id from the record path
        #raw_image_path = os.path.join(raw_images_folder, record + '.png')
        
        # load raw image
        record_image = os.path.join(raw_images_folder, record)
        image = helper_code.load_images(record_image)
        # resize image if needed
        image = resize_images(image)
        
        # get and save the gridsize
        grayscale_image = preprocessing.cepstrum_grid_detection.image_to_grayscale_array(image)
        
        # TODO: fix get_rotation_angle - it breaks for tiny_test/hr_gt/01017_hr
        rot_angle, gridsize = preprocessing.cepstrum_grid_detection.get_rotation_angle(grayscale_image)
        team_helper_code.save_gridsize(record_image, gridsize)
        
        # TODO: decide if we want to do rotation now or after the u-net. If we do it now, set image to the rotated image

        # TODO save processed image
        processed_image = os.path.join(processed_images_folder, record + '.png')
        image[0].save(processed_image,"PNG")
        

def generate_unet_training_data(wfdb_records_folder, images_folder, masks_folder, patch_folder,
                                verbose, patch_size=constants.PATCH_SIZE, records_to_process=None,
                                delete_images=False):
    """TODO
    Call generate_images_from_wfdb to generate images and masks; then patchify the images and masks
    for training the U-Net model. Save the patches in patches_folder. Option to delete images and 
    masks (not the patches) after patchifying to save space.
    """
    if not records_to_process:
        records_to_process = os.listdir(wfdb_records_folder)

    # TODO: generate images and masks

    # TODO: preprocess images (if needed)

    # generate patches
    image_patch_folder = os.path.join(patch_folder, 'image_patches')
    mask_patch_folder = os.path.join(patch_folder, 'label_patches')
    Unet.patching.save_patches_batch(images_folder, masks_folder, constants.PATCH_SIZE, 
                                     patch_folder, verbose, max_samples=False)


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

    # TODO train_unet should return trained model
    Unet.train_unet(record_ids, image_patch_folder, mask_patch_folder, args,
            PATH_UNET, CHK_PATH_UNET, LOSS_PATH, LOAD_PATH_UNET, verbose,
            max_samples=max_train_samples,
            )
    
    if delete_patches: 
        for im in os.listdir(image_patch_folder):
            os.remove(os.path.join(image_patch_folder, im))
        for im in os.listdir(mask_patch_folder):
            os.remove(os.path.join(mask_patch_folder, im))


def reconstruct_signal(record, unet_output_folder, wfdb_headers_folder, 
                       reconstructed_signals_folder, save_signal=True):
    """
    
    """
    # load header file to save with reconstructed signal
    record_path = os.path.join(wfdb_headers_folder, record) 
    header_txt = helper_code.load_header(record_path)

    # TODO: get gridsize from header file
    # alternately can pass original image in as an argument to this function and extract
    # gridsize from the image here
    ###### FIXME hardcoded gridsize for now ######
    gridsize = 37.5

    # load u-net outputs
    record_id = record.split('_')[0]
    unet_image_path = os.path.join(unet_output_folder, record_id + '.npy')
    with open(unet_image_path, 'rb') as f:
        unet_image = np.load(f)

    # reconstruct signals from u-net outputs
    reconstructed_signal, trace = ECGminer.digitize_image_unet(unet_image, gridsize, sig_len=1000)
    reconstructed_signal = np.asarray(np.nan_to_num(reconstructed_signal)) # removed *1000 astype int16

    # save reconstructed signal and copied header file in the same folder
    if save_signal:
        output_record_path = os.path.join(reconstructed_signals_folder, record)
        helper_code.save_header(output_record_path, header_txt)
        comments = [l for l in header_txt.split('\n') if l.startswith('#')]
        helper_code.save_signals(output_record_path, reconstructed_signal, comments)

    # TODO: adapt self.postprocessor.postprocess to work for different layouts

    return reconstructed_signal, trace


def generate_and_predict_unet_batch(wfdb_records_folder, images_folder, mask_folder, patch_folder,
                                  unet_output_folder, model_folder, reconstructed_signals_folder,
                                  verbose, records_to_process=None, delete_images=True):
    """
    An all-in-one to generate images from records, run them through the U-Net model, and 
    reconstruct the patches to a full image. Assumes we are generating these images, so we have 
    masks (labels), and can return a DICE score for evaluation.

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
    
    # TODO: pass in model to predict images (right now it's hardcoded in batch_predict_full_images)

    # generate patches
    Unet.patching.save_patches_batch(images_folder, mask_folder, constants.PATCH_SIZE, 
                                     patch_folder, verbose, max_samples=False)
    Unet.batch_predict_full_images(records_to_process, patch_folder, model_folder, 
                                   unet_output_folder, verbose, save_all=True)

    # reconstruct_signals
    reconstructed_signals = []
    for record in tqdm(records_to_process, desc='Reconstructing signals from U-net outputs', 
                       disable=not verbose):
        rec_signal, _ = reconstruct_signal(record, unet_output_folder, wfdb_records_folder, 
                       reconstructed_signals_folder)
        reconstructed_signals.append(rec_signal)     
        # TODO: calculate and return DICE score

    # delete patches (we have the full images/masks in images_folder)
    im_patch_dir = os.path.join(patch_folder, 'image_patches')
    label_patch_dir = os.path.join(patch_folder, 'label_patches')
    for im in os.listdir(im_patch_dir):
        os.remove(os.path.join(im_patch_dir, im))
    for im in os.listdir(label_patch_dir):
        os.remove(os.path.join(label_patch_dir, im))

    # optional: delete training images and masks, and u-net outputs
    if delete_images:
        for im in os.listdir(images_folder):
            os.remove(os.path.join(images_folder, im))
        for im in os.listdir(mask_folder):
            os.remove(os.path.join(mask_folder, im))
        for im in os.listdir(unet_output_folder):
            os.remove(os.path.join(unet_output_folder, im))


def train_classifier(reconstructed_records_folder, verbose, 
                     records_to_process=None):
    """
    Extracts features and labels from headers, then one-hot encodes labels and trains the
    SE-ResNet model.
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(reconstructed_records_folder)

    all_data = []
    labels = []
    for record in tqdm(records_to_process, desc='Loading classifier training data', 
                       disable=not verbose):
        data, label = classification.get_training_data(record, 
                                                    reconstructed_records_folder
                                                    )
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

    # TODO: check if frequency is used/if it's important
    if verbose:
        print("Training SE-ResNet classification model...")
    resnet_model = seresnet18.train_model(
                                all_data, multilabels, uniq_labels, verbose, epochs=5, 
                                validate=False
                                )
    
    if verbose:
        print("Finished training classification model.")
    
    return resnet_model, uniq_labels


def unet_predict_single_image(record_id, image, patch_folder, model, reconstructed_signal_folder,
                              verbose, delete_patches=True):
    """
    params
        record_id: str, for saving
        
    """
    # get image from image_path


    # preprocess image

    # patchify image
    # TODO will need to save/load patch size and original image size for persistence

    Unet.patching.save_patches_single_image(record_id, image, None, 
                                            patch_size=constants.PATCH_SIZE,
                                            im_patch_save_path=patch_folder,
                                            label_patch_save_path=None)

    # predict on patches
    predicted_image = Unet.predict_single_image(record_id, patch_folder, model)

    # recover u-net output image from patches

    # reconstruct signal from u-net output image

    # optional: save reconstructed signal

    # optional: delete patches

    # return reconstructed signal


def classify_signals(record, data_folder, resnet_model, classes, verbose):
    # wrap in list to match training data format
    data = [classification.get_testing_data(record, data_folder)] 
    pred_dx, probabilities = seresnet18.predict_proba(
                                        resnet_model, data, classes, verbose)
    labels = classes[np.where(pred_dx == 1)]
    if verbose:
        print(f"Classes: {classes}, probabilities: {probabilities}")
        print(f"Predicted labels: {labels}")

    return labels

    

