import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold # For multilabel stratification

from utils import default_models
from utils.utils import save_models
from team_code import train_dx_model_team, train_digitization_model_team, \
                        run_dx_model, run_digitization_model, \
                        load_dx_model, load_digitization_model


from helper_code import *

def train_dx_model(data_folder, 
            record_ids, 
            model_folder, 
            verbose, 
            models_to_train=default_models.DX_MODELS):
    """
    Parameters:
        data_folder (str): The path to the foldder containing the data.
        record_ids (list): The list of record ids, e.g. ['00001_lr', ...]
        model_folder (str): The path to the folder where the models will be saved.
        verbose (bool): Printouts?
        models_to_train (list, default: "all"): A list of the models to train, used mainly for 
            modular testing. Allows the user to specify which models should be trained. Default 
            behaviour is to train all models listed in default_models. 

    """
    if verbose:
        print('Training the dx classification model...')

    # Main function call. Pass in names of records here for cross-validation.
    # DO NOT replace the argument to models_to_train with your model. 
    # Add it to the default_models.py file instead.
    models = train_dx_model_team(data_folder, record_ids, verbose, 
                                 models_to_train=models_to_train)

    # Save the model.
    save_models(models, model_folder, verbose)
    
    if verbose:
        print('Done.')
        print()

    return models

def train_digit_model(data_folder, 
            record_ids, 
            model_folder, 
            verbose, 
            models_to_train=default_models.DIGITIZATION_MODELS):
    """
    Parameters:
        data_folder (str): The path to the foldder containing the data.
        record_ids (list): The list of record ids, e.g. ['00001_lr', ...]
        model_folder (str): The path to the folder where the models will be saved.
        verbose (bool): Printouts?
        models_to_train (list, default: "all"): A list of the models to train, used mainly for 
            modular testing. Allows the user to specify which models should be trained. Default 
            behaviour is to train all models listed in default_models. 

    """
    # Find data files.
    if verbose:
        print('Training the digitization model...')

    # Main function call. Pass in names of records here for cross-validation.
    models = train_digitization_model_team(data_folder, record_ids, verbose,
                                           models_to_train=models_to_train)

    # Save the model.
    save_models(models, model_folder, verbose)

    if verbose:
        print('Done.')
        print()

    return models

def run(data_folder, 
        records,
        output_folder, 
        digit_model, 
        dx_model,
        allow_failures=False, verbose=True):
    """
    Parameters:
        dx_model (dict): The trained model.
        record (str): The path to the record to classify.
        signal (np.ndarray): The signal to classify.
        verbose (bool): printouts? you want 'em, we got 'em
    """

    num_records = len(records)
    if num_records==0:
        raise Exception('No data were provided.')

    # Run the team's model(s) on the Challenge data.
    if verbose:
        print('Running the Challenge model(s) on the Challenge data...')

    predictions = []

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        data_record = os.path.join(data_folder, records[i])
        output_record = os.path.join(output_folder, records[i])

        # Run the digitization model. Allow or disallow the model to fail on some of the data, which can be helpful for debugging.
        try:
            signal = run_digitization_model(digit_model, data_record, verbose) ### Teams: Implement this function!!!
        except:
            if allow_failures:
                if verbose:
                    print('... digitization failed.')
                signal = None
            else:
                raise

        # Run the dx classification model. Allow or disallow the model to fail on some of the data, which can be helpful for debugging.
        try:
            dx = run_dx_model(dx_model, data_record, signal, verbose) ### Teams: Implement this function!!!
        except:
            if allow_failures:
                if verbose >= 2:
                    print('... dx classification failed.')
                dx = None
            else:
                raise

        predictions.append(dx)
        
        # Save Challenge outputs.
        output_path = os.path.split(output_record)[0]
        os.makedirs(output_path, exist_ok=True)

        data_header = load_header(data_record)
        save_header(output_record, data_header)

        if signal is not None:
            save_signal(output_record, signal)

            comment_lines = [l for l in data_header.split('\n') if l.startswith('#')]
            signal_header = load_header(output_record)
            signal_header += ''.join(comment_lines) + '\n'
            save_header(output_record, signal_header)

        if dx is not None:
            save_dx(output_record, dx)

    if verbose:
        print('Done.')

    return predictions

def five_fold_cv(data_folder=None, multilabel_cv=True, n_splits=5, train_models=True):

    if data_folder is None:
        data_folder = os.path.join('tiny_testset', 'records100')

    # Find the records
    records = find_records(data_folder)
    if len(records) == 0:
        raise FileNotFoundError('No data was provided.')

    # Find the labels
    labels = []
    for r in records:
        currect_dx = load_dx(os.path.join(data_folder, r))
        labels.append(currect_dx)

    if not labels:
        raise Exception('There are no labels for the data.')
    
    ohe = OneHotEncoder(sparse_output=False)
    labels = ohe.fit_transform(labels)
    uniq_labels = ohe.categories_[0]

    # Initialize either multilabel stratified K-fold or standard stratified K-fold
    kfold = MultilabelStratifiedKFold(n_splits = n_splits) if multilabel_cv else StratifiedKFold(n_splits=n_splits)
    indeces = np.arange(len(records))

    result_str = datetime.now().strftime("%m/%d/%Y") + f" Test digitalization models: {default_models.DIGITIZATION_MODELS} \t Test dx models: {default_models.DX_MODELS} \n"
    # Iterate over CV splits
    for i, (train_idx, test_idx) in enumerate(kfold.split(indeces, labels)):
        train_records, test_records = list(map(records.__getitem__, train_idx)), list(map(records.__getitem__, test_idx))
        
        model_folder = os.path.join('cv_results', 'trained models', f'split_{i+1}')
        os.makedirs(model_folder, exist_ok=True)
        
        output_folder = os.path.join('cv_results','output_folder', f'split_{i+1}')
        os.makedirs(output_folder, exist_ok=True)

        team_digit_model = train_digit_model(data_folder, train_records, model_folder, True)
        team_dx_model = train_dx_model(data_folder, train_records, model_folder, True)
        preds = run(data_folder, test_records, output_folder, team_digit_model, team_dx_model,
                    allow_failures=False, verbose=True)

        # TBA: EVALUATION
