#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you 
# can edit most parts of the required functions, change or remove non-required 
# functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os, time, joblib, sys
import numpy as np
from tqdm import tqdm
import traceback

import helper_code
import preprocessing, reconstruction, classification, reconstruction.image_cleaning
from utils import default_models, utils, team_helper_code
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the 
# arguments of the functions.
#
################################################################################

# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    """
    A wrapper function for training the digitization model. Loads in the data files,
    calls train_digitization_model_team() to train the model, and saves it to the 
    model_folder.
    NOTE: Do not edit the arguments of this function.
    """

    # Find data files.
    if verbose:
        print('Training the digitization model...')
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    if len(records) == 0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Main function call. Pass in names of records here for cross-validation.
    models = train_digitization_model_team(data_folder, records, verbose)

    # Save the model.
    utils.save_models(models, model_folder, verbose)

    if verbose:
        print('Done.')
        print()


# Train your dx model.
def train_dx_model(data_folder, model_folder, verbose):
    """
    A wrapper function for training the dx classification model. Loads in the data files,
    calls train_dx_model_team() to train the model, and saves it to the model_folder.
    NOTE: Do not edit the arguments of this function.
    """

    # Find data files.
    if verbose:
        print('Training the dx classification model...')
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    if len(records) == 0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Main function call. Pass in names of records here for cross-validation.
    # DO NOT replace the argument to models_to_train with your model. 
    # Add it to the default_models.py file instead.
    models = train_dx_model_team(data_folder, records, verbose, 
                                 models_to_train=default_models.DX_MODELS
                                 )

    # Save the model.
    utils.save_models(models, model_folder, verbose)

    if verbose:
        print('Done.')
        print()


# Load your trained digitization model. This function is *required*. You should edit 
# this function to add your code, but do *not* change the arguments of this function.
# If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    models = utils.load_models(model_folder, verbose, default_models.DIGITIZATION_MODELS)
    return models


# Load your trained dx classification model. This function is *required*. You should 
# edit this function to add your code, but do *not* change the arguments of this 
# function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder, verbose):
    models_to_load = default_models.DX_MODELS + ['dx_classes']
    models = utils.load_models(model_folder, verbose, models_to_load)
    return models


# Run your trained digitization model. This function is *required*. You should edit 
# this function to add your code, but do *not* change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):

    image = helper_code.load_image(record)
    header_file = helper_code.get_header_file(record)
    header = helper_code.load_text(header_file)

    num_samples = helper_code.get_num_samples(header)
    num_signals = helper_code.get_num_signals(header)
    signal=None

    ###### Example model ######
    if 'digit_example' in digitization_model.keys():
        
        model = digitization_model['digit_example']
        # Extract features.
        features = preprocessing.example.extract_features(record)

        # For a overly simply minimal working example, generate "random" waveforms.
        seed = int(round(model + np.mean(features)))
        signal = np.random.default_rng(seed=seed).uniform(low=-1000, 
                                                            high=1000, 
                                                            size=(num_samples, num_signals))

    if 'digit_clean_miner' in digitization_model.keys():
        image_files = team_helper_code.load_image_paths(record)
        image_file = image_files[0]
        if len(image_files) > 1:
            if verbose:
                print(f"Multiple images found, using image at {image_file}.")
        try:
            signal = reconstruction.image_cleaning.digitize(image_file)
            signal = np.nan_to_num(signal)
        except Exception as e: 
            # FIXME for now, let the code continue if there's an error
            if verbose:
                print(f"Error digitizing image {image_file}: {e}")
                # print(traceback.format_exc())
    
    if signal is not None:
        try:
            signal = np.asarray(signal, dtype=np.int16)
        except ValueError:
            raise ValueError("Could not digitalize signal. Check that you've loaded the right model(s).")

    return signal


# Run your trained dx classification model. This function is *required*. You should edit 
# this function to add your code, but do *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):
    """
    Parameters:
        dx_model (dict): The trained model.
        record (str): The path to the record to classify.
        signal (np.ndarray): The signal to classify.
        verbose (bool): printouts? you want 'em, we got 'em
    """

    classes = dx_model['dx_classes']

    ####### Example model ########
    if 'dx_example' in dx_model.keys():
        # Extract features.
        features = preprocessing.demographics.extract_features(record) # for consistency with train
        features = features.reshape(1, -1)

        # Get model probabilities.
        model = dx_model['dx_example']
        probabilities = model.predict_proba(features)
        probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

        # Choose the class(es) with the highest probability as the label(s).
        max_probability = np.nanmax(probabilities)   
        labels = [classes[i] for i, probability in enumerate(probabilities) if 
              probability == max_probability]

    ######### SEResNet ###########
    if 'seresnet' in dx_model.keys():
        # Extract features: load header
        header = helper_code.load_header(record)
        try:
            age_gender = preprocessing.demographics.extract_features(record)
        except ValueError as e:
            if verbose:
                print(f"No demographic data found for record {record}.")
            age_gender = np.zeros(3)
        try:
            fs = helper_code.get_sampling_frequency(header)
        except ValueError as e:
            if verbose:
                print(f"No sampling frequency found for record {record}.")
            fs = 100
        
        # if signal is None, attempt to load from file
        if signal is None:
            if verbose: 
                print(f"Signal is None for record {record}. Attempting to load from file...")
            try:
                signal, _ = helper_code.load_signal(record)
            except FileNotFoundError:
                # FIXME this will continue to run if reconstruction fails
                print(f"No signal or signal file found for record {record}. ")
                num_samples = helper_code.get_num_samples(header)
                num_signals = helper_code.get_num_signals(header)
                signal = np.zeros((num_samples, num_signals))

        data = [[signal, fs, age_gender]]

        # Get model probabilities.
        model = dx_model['seresnet']
        # mutli_dx_threshold is probability above which secondary labels are returned positive in pred_dx
        pred_dx, probabilities = classification.seresnet18.predict_proba(
                                        model, data, classes, verbose, multi_dx_threshold=0.5)
        labels = classes[np.where(pred_dx == 1)]
        if verbose:
            print(f"Classes: {classes}, probabilities: {probabilities}")
            print(f"Predicted labels: {labels}")

    try:
        if len(labels) == 0:
            raise ValueError("No or invalid probabilities returned. "+
                         "Check that your model can cast probabilities to labels correctly.")
    except UnboundLocalError:
        raise UnboundLocalError("No labels returned. "+
                         "Check that you've loaded the right model(s).")
    return labels


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def train_digitization_model_team(data_folder, records, verbose, 
                                  models_to_train=default_models.DIGITIZATION_MODELS):
    """
    Main function call for train_digitization_model(). 

    Parameters:
        data_folder (str): The path to the foldder containing the data.
        records (list): A list of the records to use for training the model. e.g. ['00001_lr']
        verbose (bool): How many printouts do you want?
        models_to_train (list, default: "all"): A list of the models to train, used mainly for 
            modular testing. Allows the user to specify which models should be trained. Default 
            behaviour is to train all models listed in default_models. 

    Returns:
        team_models (dict): The trained models.
    """
    models = {} 

    if 'digit_clean_miner' in models_to_train:
        # put this at the top since we don't train anything in this case
        models['digit_clean_miner'] = -1 # return a dummy value
        if len(models_to_train) == 1: # if this is the only model we're using
            if verbose: 
                print("Performing algorithmic digitization only (no training); a dummy model will be returned.")
            return models # return without running any more code
    if 'digit_example' in models_to_train:
        # WARNING: this returns None, not a valid model
        if len(models_to_train) == 1: # if this is the only model we're using
            if verbose: 
                print("Not performing digitization; returning None.")
            return None

    start = time.time() # because I am impatient

    ############## Extract the features and labels. ###############
    if verbose:
        print('Extracting features and labels from the data...')
        t1 = time.time()

    num_records = len(records)
    features = list()

    for i in tqdm(range(num_records), disable=~verbose):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
    
        record = os.path.join(data_folder, records[i])

        # Extract the features from the image...
        current_features = preprocessing.example.extract_features(record)
        features.append(current_features)

    if verbose:
        t2 = time.time()
        print(f'Done. Time to extract features: {t2 - t1:.2f} seconds.')

    ############## Train the models. ################
    if verbose:
        print('Training the model on the data...')

    if 'digit_example' in models_to_train:
        models['digit_example'] = reconstruction.example.train(features)

    if verbose:
        print(f'Done. Time to train individual models: {time.time() - t2:.2f} seconds.')
        print(f'Total time elapsed: {time.time() - start:.2f} seconds.')

    return models


def train_dx_model_team(data_folder, records, verbose, 
                        models_to_train=default_models.DX_MODELS):
    """
    Main function call for train_dx_model().
    
    Parameters:
        data_folder (str): The path to the foldder containing the data.
        records (list): A list of the records to use for training the model. e.g. ['00001_lr']
        verbose (bool): How many printouts do you want?
        models_to_train (list, default: "all"): A list of the models to train, used mainly for 
            modular testing. Allows the user to specify which models should be trained. Default 
            behaviour is to train all models listed in default_models. 

    Returns:
        team_models (dict): The trained models.
    """
    start = time.time() # because I am impatient
    models = {} 

    ############## Extract the features and labels. ###############
    if verbose:
        print('Extracting features and labels from the data...')
        t1 = time.time()

    num_records = len(records)
    record_paths = []
    features = list()
    labels = list()
    fs_arr = list()

    # Iterate over recordings and find: FS, AGE, SEX
    for i in tqdm(range(num_records), disable=~verbose):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image, but only if the image has one or more dx classes.
        dx = helper_code.load_dx(record)
        if dx and dx != [""]:
            # age_gender is len 3 array: (age/100, male, female)
            age_gender = preprocessing.demographics.extract_features(record) 
            features.append(age_gender) 
            labels.append(dx)
            
            # Load header
            header = helper_code.load_header(record)
            fs = helper_code.get_sampling_frequency(header)
            fs_arr.append(fs)

            # current_features = preprocessing.example.extract_features(record)
            # features.append(current_features)

            record_paths.append(record)

    if not labels:
        raise Exception('There are no labels for the data.')  
    
    # ========= Combine data obtained =====
    # Take order of variables into account
    data = [list(ls) for ls in zip(record_paths, fs_arr, features)] 
    # =====================================

    # We don't need one hot encoding?
    # One-hot-encode labels 
    mlb = MultiLabelBinarizer()
    multilabels = mlb.fit_transform(labels)
    uniq_labels = mlb.classes_
    models['dx_classes'] = uniq_labels

    if verbose:
        t2 = time.time()
        print(f'Done. Time to extract features and labels: {t2 - t1:.2f} seconds.')

    ############## Train the models. ################
    if verbose:
        print('Training the model on the data...')

    if 'dx_example' in models_to_train:
        models['dx_example'] = classification.example.train(features, labels)

    if 'seresnet' in models_to_train:
        models['seresnet'] = classification.seresnet18.train_model(
                                    data, multilabels, uniq_labels, verbose, epochs=5, validate=False
                                )

    if verbose:
        print(f'Done. Time to train individual models: {time.time() - t2:.2f} seconds.')
        print(f'Total time elapsed: {time.time() - start:.2f} seconds.')

    return models
