import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
from datetime import datetime
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold # For multilabel stratification

from utils import default_models, utils
import team_code

import helper_code


def generate_images(signal_folder, output_folder, verbose=True):
    raise NotImplementedError


def train_digit_model(images_folder, 
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
    models = train_digitization_model_team(images_folder, record_ids, verbose,
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

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        data_record = os.path.join(data_folder, records[i])
        output_record = os.path.join(output_folder, records[i])

        # Run the digitization model. Allow or disallow the model to fail on some of the data, which can be helpful for debugging.
        try:
            signal = team_code.run_digitization_model(digit_model, data_record, verbose)
        except Exception as e:
            if allow_failures:
                if verbose:
                    print(f'... digitization failed on {records[i]} with {e}.')
                signal = None
            else:
                raise

        # Save Challenge outputs.
        output_path = os.path.split(output_record)[0]
        os.makedirs(output_path, exist_ok=True)

        data_header = helper_code.load_header(data_record)
        helper_code.save_header(output_record, data_header)

        if signal is not None:
            helper_code.save_signal(output_record, signal)

            comment_lines = [l for l in data_header.split('\n') if l.startswith('#')]
            signal_header = helper_code.load_header(output_record)
            signal_header += ''.join(comment_lines) + '\n'
            helper_code.save_header(output_record, signal_header)

    if verbose:
        print('Done.')
        

def evaluate(label_folder, records, output_folder, extra_scores = True):
    # Compute the signal reconstruction metrics.
    channels = list()
    records_completed_signal_reconstruction = list()
    snrs = list()
    snrs_median = list()
    ks_metric = list()
    asci_metric = list()
    weighted_absolute_difference_metric = list()

    # Iterate over the records.
    for record in records:
        # Load the signals, if available.
        label_record = os.path.join(label_folder, record)
        label_signal, label_fields = helper_code.load_signal(label_record)

        if label_signal is not None:
            label_channels = label_fields['sig_name']
            label_sampling_frequency = label_fields['fs']
            label_num_samples = label_fields['sig_len']
            channels.append(label_channels) # Use this variable if computing aggregate statistics for each channel.

            output_record = os.path.join(output_folder, record)
            output_signal, output_fields = helper_code.load_signal(output_record)

            if output_signal is not None:
                output_channels = output_fields['sig_name']
                output_sampling_frequency = output_fields['fs']                
                output_num_samples = output_fields['sig_len']
                records_completed_signal_reconstruction.append(record)

                ###
                ### TO-DO: Perform checks, such as sampling frequency, units, etc.
                ###

                # Reorder and trim or pad the signal as needed.
                output_signal = helper_code.reorder_signal(output_signal, output_channels, label_channels)
                output_signal = helper_code.trim_signal(output_signal, label_num_samples)

            else:
                output_signal = np.zeros(np.shape(label_signal), dtype=label_signal.dtype)

            # Compute the signal reconstruction metrics.
            channels = label_channels
            sampling_frequency = label_sampling_frequency
            num_channels = len(label_channels)

            values = list()
            for j in range(num_channels):
                value = helper_code.compute_snr(label_signal[:, j], output_signal[:, j])
                values.append(value)
            snrs.append(values)

            if extra_scores:
                values = list()
                for j in range(num_channels):
                    value = helper_code.compute_snr_median(label_signal[:, j], output_signal[:, j])
                    values.append(value)
                snrs_median.append(values) 

                values = list()
                for j in range(num_channels):
                    value = helper_code.compute_ks_metric(label_signal[:, j], output_signal[:, j])
                    values.append(value)
                ks_metric.append(values)

                values = list()
                for j in range(num_channels):
                    value = helper_code.compute_asci_metric(label_signal[:, j], output_signal[:, j])
                    values.append(value)
                asci_metric.append(values)             
    
                values = list()
                for j in range(num_channels):
                    value = helper_code.compute_weighted_absolute_difference(label_signal[:, j], output_signal[:, j], sampling_frequency)
                    values.append(value)
                weighted_absolute_difference_metric.append(values)

    if records_completed_signal_reconstruction:
        snrs = np.concatenate(snrs)
        mean_snr = np.nanmean(snrs)

        if extra_scores:
            snrs_median = np.concatenate(snrs_median)
            mean_snr_median = np.nanmean(snrs_median)

            ks_metric = np.concatenate(ks_metric)
            mean_ks_metric = np.nanmean(ks_metric)

            asci_metric = np.concatenate(asci_metric)
            mean_asci_metric = np.nanmean(asci_metric)

            weighted_absolute_difference_metric = np.concatenate(weighted_absolute_difference_metric)
            mean_weighted_absolute_difference_metric = np.nanmean(weighted_absolute_difference_metric)
        else:
            mean_snr_median = float('nan')
            mean_ks_metric = float('nan')
            mean_asci_metric = float('nan')     
            mean_weighted_absolute_difference_metric = float('nan')          

    else:
        mean_snr = float('nan')
        mean_snr_median = float('nan')
        mean_ks_metric = float('nan')
        mean_asci_metric = float('nan')  
        mean_weighted_absolute_difference_metric = float('nan')          

    # Compute the classification metrics.
    records_completed_classification = list()
    label_dxs = list()
    output_dxs = list()

    # Iterate over the records.
    for record in records:
        # Load the classes, if available.
        label_record = os.path.join(label_folder, record)
        label_dx = helper_code.load_dx(label_record)

        if label_dx:
            output_record = os.path.join(output_folder, record)
            output_dx = helper_code.load_dx(output_record)

            if output_dx:
                records_completed_classification.append(record)

            label_dxs.append(label_dx)
            output_dxs.append(output_dx)

    if records_completed_classification:
        f_measure, _, _ = helper_code.compute_f_measure(label_dxs, output_dxs)
    else:
        f_measure = float('nan')

    # Return the results.
    return mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, mean_weighted_absolute_difference_metric, f_measure


def five_fold_cv(data_folder=None, images_folder=None, generate_images=False, verbose=True):
    result_str = datetime.now().strftime("%m/%d/%Y") + f" Test digitalization models: {default_models.DIGITIZATION_MODELS}\n"

    if generate_images:
        if data_folder is None:
            try:
                data_folder = os.path.join('..', 'ptb-xl', 'records500', '00000')
            except FileNotFoundError:
                raise FileNotFoundError('No data was provided. Please provide a path to the data folder.')        
        # pass in results string so we can print out parameters used to generate images
        images_folder, result_str = generate_images(data_folder, images_folder, result_str, verbose=True)

    elif images_folder is None:
        try:
            images_folder = os.path.join('..', 'ptb-xl', 'img_records500', '00000')
        except FileNotFoundError:
            raise FileNotFoundError("No data was provided. Please provide a path to the folder containing generated "+\
                                    "images, or use argument --generate_images True.")
    else: # not generating images, images_folder is provided
        print(f"Looking for images in {images_folder}...")   

    
    # Iterate over CV splits
    for i, (train_idx, test_idx) in enumerate(kfold.split(indeces, labels)):
        train_records, test_records = list(map(records.__getitem__, train_idx)), list(map(records.__getitem__, test_idx))
        result_str += f'========== SPLIT {i+1} ========== \n'

        # Initialize the directories where to store the models and the outputs
        model_folder = os.path.join('cv_results', 'trained models', f'split_{i+1}')
        os.makedirs(model_folder, exist_ok=True)
        
        output_folder = os.path.join('cv_results','output_folder', f'split_{i+1}')
        os.makedirs(output_folder, exist_ok=True)

        # First, train digitalization model and classifier and then run them
        team_digit_model = train_digit_model(data_folder, train_records, model_folder, True)

        run(data_folder, test_records, output_folder, team_digit_model,
            allow_failures=False, verbose=True)

        # Evaluate the outputs
        scores = evaluate(data_folder, test_records, output_folder)
        snr, snr_median, ks_metric, asci_metric, mean_weighted_absolute_difference_metric, f_measure = scores
        result_str += \
            f'SNR: {snr:.3f}\n' + \
            f'SNR median: {snr_median:.3f}\n' \
            f'KS metric: {ks_metric:.3f}\n' + \
            f'ASCI metric: {asci_metric:.3f}\n' \
            f'Weighted absolute difference metric: {mean_weighted_absolute_difference_metric:.3f}\n' \
            f'F-measure: {f_measure:.3f}\n\n'
    
    # Store the results
    save_path = os.path.join('cv_results', 'results_' + datetime.now().strftime("%Y%m%d-%H%M") + '.txt')
    with open(save_path, 'w') as f:
         f.write(result_str)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-i', '--images_folder', type=str, required=False)
    parser.add_argument('-g', '--generate_images', type=bool, default=False)
    parser.add_argument('-v', '--verbose', type=bool, default=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    five_fold_cv(args)