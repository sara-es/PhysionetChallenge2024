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

import helper_code 
import preprocessing, reconstruction, classification
from utils import default_models, utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold # For multilabel stratification
from classification.train_utils import Training

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
    models = train_dx_model_team(data_folder, records, verbose, models_to_train='seresnet')

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
    # Extract features.
    features = preprocessing.example.extract_features(record)

    # Load the dimensions of the signal.
    header_file = helper_code.get_header_file(record)
    header = helper_code.load_text(header_file)

    num_samples = helper_code.get_num_samples(header)
    num_signals = helper_code.get_num_signals(header)

    if 'digit_example' in digitization_model.keys():
        model = digitization_model['digit_example']

        # For a overly simply minimal working example, generate "random" waveforms.
        seed = int(round(model + np.mean(features)))
        signal = np.random.default_rng(seed=seed).uniform(low=-1000, 
                                                            high=1000, 
                                                            size=(num_samples, num_signals))
    try:
        signal = np.asarray(signal, dtype=np.int16)
    except ValueError:
        raise ValueError("Could not digitalize signal. Check that you've loaded the right model(s).")

    return signal


# Run your trained dx classification model. This function is *required*. You should edit 
# this function to add your code, but do *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):
    classes = dx_model['dx_classes']

    # Extract features.
    features = preprocessing.example.extract_features(record)
    features = features.reshape(1, -1)

    # Get model probabilities.
    if 'dx_example' in dx_model.keys():
        model = dx_model['dx_example']
        probabilities = model.predict_proba(features)
        probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    # Choose the class(es) with the highest probability as the label(s).
    try: 
        max_probability = np.nanmax(probabilities)
    except ValueError:
        raise ValueError("No probabilities returned. Check that you've loaded the right model(s).")
    
    labels = [classes[i] for i, probability in enumerate(probabilities) if 
              probability == max_probability]

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
    start = time.time() # because I am impatient
    models = {} 

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
        record_paths.append(record)

        # Extract the features from the image, but only if the image has one or more dx classes.
        dx = helper_code.load_dx(record)
        if dx:
            age_gender = preprocessing.demographics.extract_features(record) # len 3 array (age/100, male, female)
            features.append(age_gender) # => splitted the ag array just for simplicity (for now)
            labels.append(dx)
            
            # Load header
            header = helper_code.load_header(record)
            fs = helper_code.get_sampling_frequency(header)
            fs_arr.append(fs)

            # current_features = preprocessing.example.extract_features(record)
            # features.append(current_features)
            
    if not labels:
        raise Exception('There are no labels for the data.')  
    
    # ========= Combine data obtained =====
    data = [list(ls) for ls in zip(record_paths, fs_arr, features)] # Take order of variables into account
    # =====================================

    # We don't need one hot encoding?
    # One-hot-encode labels 
    ohe = OneHotEncoder(sparse_output=False)
    multilabels = ohe.fit_transform(labels)
    uniq_labels = ohe.categories_[0] # order of the labels!
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
        # This is now set in the ECGDataset class so no need to set it here unless we choose otherwise :)
        #channels = 12 # TODO: MAGIC NUMBER find a way to get the number of channels from the data
        
        # ============= VALIDATION? =============
        perform_validation = True ## WHERE TO SET THIS?

        if perform_validation:
            # 1) Split data to training and validation; return indeces for training and validation sets
            # Either one stratified train/val split OR Stratified K-fold
            split_index_list = split_data(data, multilabels, n_splits=5) # Default, one train/val split

            # Iterate over train/test splits
            pool_metrics = []
            for train_idx, val_idx in split_index_list:
                train_data, val_data = list(map(data.__getitem__, train_idx)), list(map(data.__getitem__, val_idx))
                train_labels, val_labels = list(map(multilabels.__getitem__, train_idx)), list(map(multilabels.__getitem__, val_idx))

                args = {'train_data': train_data, 'val_data': val_data,
                        'train_labels': train_labels, 'val_labels': val_labels,
                        'dx_labels': uniq_labels, 'epochs': 5, 'batch_size': 5}
                
                # 2) Training ResNet model(s) on the training data and evaluating on the validation set
                trainer = Training(args)
                trainer.setup()
                metrics = trainer.train(compute_metrics=True) # Compute also the classification metrics (now, F-measure)
                pool_metrics.append(metrics)  

            print('\nValidation phase performed using {}'.format('basic train/val split' 
                                                               if len(split_index_list) == 1 
                                                               else '{}-Fold'.format(len(split_index_list ))))
            print('\t - F-measure: {}'.format(pool_metrics[0] 
                                              if len(split_index_list) == 1 
                                              else np.nanmean(pool_metrics)))
        
        else: # Only train the model

            # Train the model using entire data and store the state dictionary
            args = {'train_data': data, 'val_data': None,
                    'train_labels': multilabels, 'val_labels': None,
                    'dx_labels': uniq_labels, 'epochs': 5, 'batch_size': 5}
            
            trainer = Training(args)
            trainer.setup()
            models['state_dict'] = trainer.train() 
            return models

    if verbose:
        print(f'Done. Time to train individual models: {time.time() - t2:.2f} seconds.')
        print(f'Total time elapsed: {time.time() - start:.2f} seconds.')

    return models

# =======================================================================
# Multilabel version
# Splitting data into two sets based on number of splits that are needed
# return indeces of the data for the splits
def split_data(data, labels, n_splits=1):
    idx = np.arange(len(data))
    split_index_list = []

    if n_splits == 1: # One train/Test split
        mss = MultilabelStratifiedShuffleSplit(n_splits = n_splits, train_size=.75, test_size=.25, random_state=2024)
        for train_idx, test_idx in mss.split(idx, labels):
            split_index_list.append([train_idx, test_idx])
        
    else: # K-Fold
        skfold = MultilabelStratifiedKFold(n_splits = n_splits)
        for train_idx, test_idx in skfold.split(idx, labels):
            split_index_list.append([train_idx, test_idx])

    return split_index_list

# Binary version
def _split_data(data, labels, n_splits=1):
    idx = np.arange(len(data))
    split_indeces = []

    if n_splits == 1: # Basic train/test split
        train_idx, test_idx = train_test_split(idx, stratify=labels, test_size=.25, random_state=2024)
        split_indeces.append([train_idx, test_idx])
    
    else: # K-Fold
        skfold = StratifiedKFold(n_splits=n_splits)
        for train_idx, test_idx in skfold.split(idx, labels):
            split_indeces.append([train_idx, test_idx])

    return split_indeces
