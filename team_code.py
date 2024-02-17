#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os, sys, time, joblib
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch

import helper_code 
import preprocessing, reconstruction, classification
from utils import default_models, utils

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
    calls train_digitization_model_team() to train the model, and saves it to the model_folder.
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
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            utils.save_model_torch(model, name, model_folder)
        else:
            utils.save_model_pkl(model, name, model_folder)
        if verbose >= 2:
            print(f'{name} model saved.')

    if verbose:
        print('Done.')
        print()
        

# Train your dx model.
def train_dx_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the dx classification model...')
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()
    dxs = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image, but only if the image has one or more dx classes.
        dx = load_dx(record)
        if dx:
            current_features = preprocessing.example.extract_features(record)
            features.append(current_features)
            dxs.append(dx)

    if not dxs:
        raise Exception('There are no labels for the data.')

    features = np.vstack(features)
    classes = sorted(set.union(*map(set, dxs)))
    dxs = helper_code.compute_one_hot_encoding(dxs, classes)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state   = 56  # Random state; set for reproducibility.

    # Fit the model.
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, dxs)

    # Save the model.
    save_dx_model(model_folder, model, classes)

    if verbose:
        print('Done.')
        print()

# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'digitization_model.sav')
    return joblib.load(filename)

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'dx_model.sav')
    return joblib.load(filename)

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):
    model = digitization_model['model']

    # Extract features.
    features = preprocessing.example.extract_features(record)

    # Load the dimensions of the signal.
    header_file = helper_code.get_header_file(record)
    header = helper_code.load_text(header_file)

    num_samples = helper_code.get_num_samples(header)
    num_signals = helper_code.get_num_signals(header)

    # For a overly simply minimal working example, generate "random" waveforms.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1000, high=1000, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.int16)

    return signal

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):
    model = dx_model['model']
    classes = dx_model['classes']

    # Extract features.
    features = preprocessing.example.extract_features(record)
    features = features.reshape(1, -1)

    # Get model probabilities.
    probabilities = model.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    # Choose the class(es) with the highest probability as the label(s).
    max_probability = np.nanmax(probabilities)
    labels = [classes[i] for i, probability in enumerate(probabilities) if probability == max_probability]

    return labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def train_digitization_model_team(data_folder, records, verbose, models_to_train=['all'], allow_failures=False):
    """
    Main function call for train_digitization_model(). 

    Parameters:
        data_folder (str): The path to the foldder containing the data.
        records (list): A list of the records to use for training the model. e.g. ['00001_lr']
        verbose (bool): How many printouts do you want?
        models_to_train (list, default: "all"): A list of the models to train, used mainly for modular testing.
            Allows the user to specify which models should be trained. Default behaviour is to train all models
            available. 
        allow_failures (bool, default: False): when testing, allow code to continue if a recording 
            cannot be loaded. Needs to be False for official Challenge runs.

    Returns:
        team_models (dict): The trained models.
    """

    if ['all'] in models_to_train:
        models_to_train = default_models.DIGITIZATION_MODELS

    start = time.time() # because I am impatient
    models = {} 

    ############## Extract the features and labels. ###############
    if verbose:
        print('Extracting features and labels from the data...')
        t1 = time.time()

    num_records = len(records)
    features = list()

    for i in tqdm(range(num_records)):
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

    models['example'] = reconstruction.example.train(features)

    if verbose:
        print(f'Done. Time to train individual models: {time.time() - t2:.2f} seconds.')
        print(f'Total time elapsed: {time.time() - start:.2f} seconds.')

    return models