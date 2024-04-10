import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import traceback, tqdm, argparse, time
import wfdb

import team_code, helper_code
from reconstruction import image_cleaning
from utils import team_helper_code


def run_digitization_model(image_file, verbose, allow_failures=False):
    # clean and rotate the image
    cleaned_image, gridsize = image_cleaning.clean_image(image_file)   

    # digitize with ECG-miner
    try:
        signal, trace = image_cleaning.digitize_image(cleaned_image, gridsize)
        signal = np.nan_to_num(signal)
    except Exception as e: 
        if allow_failures:
            signal = None
            if verbose:
                print(f"Error digitizing image {image_file}: {e}")
                print(traceback.format_exc())
        else: raise e

    return trace, signal, gridsize


def single_signal_snr(output_signal, output_fields, label_signal, label_fields, extra_scores=False):
    # lifted from evaluate_model.py
    snr = dict()
    snr_median = dict()
    ks_metric = dict()
    asci_metric = dict()
    weighted_absolute_difference_metric = dict()

    if label_signal is not None:
        record = output_fields['sig_ID']
        label_channels = label_fields['sig_name']
        label_num_channels = label_signal.shape[1]
        label_num_samples = label_signal.shape[0]
        label_sampling_frequency = label_fields['fs']
        label_units = label_fields['units']

        if output_signal is not None:
            output_channels = output_fields['sig_name']
            output_num_channels = output_signal.shape[1]
            output_num_samples = output_signal.shape[0]
            output_sampling_frequency = output_fields['fs']
            output_units = output_fields['units']

            # Check that the label and output signals match as expected.
            assert(label_sampling_frequency == output_sampling_frequency)
            assert(label_units == output_units)

            # Reorder the channels in the output signal to match the channels in the label signal.
            output_signal = helper_code.reorder_signal(output_signal, output_channels, label_channels)

            # Trim or pad the channels in the output signal to match the channels in the label signal.
            output_signal = helper_code.trim_signal(output_signal, label_num_samples)

            # Replace the samples with NaN values in the output signal with zeros.
            output_signal[np.isnan(output_signal)] = 0

        else:
            output_signal = np.zeros(np.shape(label_signal), dtype=label_signal.dtype)

        # Compute the signal reconstruction metrics.
        channels = label_channels
        num_channels = label_num_channels
        sampling_frequency = label_sampling_frequency

        for j, channel in enumerate(channels):
            value = helper_code.compute_snr(label_signal[:, j], output_signal[:, j])
            snr[(record, channel)] = value

            if extra_scores:
                value = helper_code.compute_snr_median(label_signal[:, j], output_signal[:, j])
                snr_median[(record, channel)] = value

                value = helper_code.compute_ks_metric(label_signal[:, j], output_signal[:, j])
                ks_metric[(record, channel)] = value

                value = helper_code.compute_asci_metric(label_signal[:, j], output_signal[:, j])
                asci_metric[(record, channel)] = value

                value = helper_code.compute_weighted_absolute_difference(label_signal[:, j], output_signal[:, j], sampling_frequency)
                weighted_absolute_difference_metric[(record, channel)] = value
    
    snr = np.array(list(snr.values()))
    if not np.all(np.isnan(snr)):
        mean_snr = np.nanmean(snr)
    else:
        mean_snr = float('nan')

    if extra_scores:
        snr_median = np.array(list(snr_median.values()))
        if not np.all(np.isnan(snr_median)):
            mean_snr_median = np.nanmean(snr_median)
        else:
            mean_snr_median = float('nan')

        ks_metric = np.array(list(ks_metric.values()))
        if not np.all(np.isnan(ks_metric)):
            mean_ks_metric = np.nanmean(ks_metric)
        else:
            mean_ks_metric = float('nan')

        asci_metric = np.array(list(asci_metric.values()))
        if not np.all(np.isnan(asci_metric)):
            mean_asci_metric = np.nanmean(asci_metric)
        else:
            mean_asci_metric = float('nan')

        weighted_absolute_difference_metric = np.array(list(weighted_absolute_difference_metric.values()))
        if not np.all(np.isnan(weighted_absolute_difference_metric)):
            mean_weighted_absolute_difference_metric = np.nanmean(weighted_absolute_difference_metric)
        else:
            mean_weighted_absolute_difference_metric = float('nan')
    else:
        mean_snr_median = float('nan')
        mean_ks_metric = float('nan')
        mean_asci_metric = float('nan')
        mean_weighted_absolute_difference_metric = float('nan')
    
    return mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, mean_weighted_absolute_difference_metric


def format_wfdb_signal(header, signal, comments=list()):
    sampling_frequency = helper_code.get_sampling_frequency(header)
    signal_formats = helper_code.get_signal_formats(header)
    adc_gains = helper_code.get_adc_gains(header)
    baselines = helper_code.get_baselines(header)
    signal_units = helper_code.get_signal_units(header)
    signal_names = helper_code.get_signal_names(header)

    if all(signal_format == '16' for signal_format in signal_formats):
        signal = np.clip(signal, -2**15 + 1, 2**15 - 1)
        signal = np.asarray(signal, dtype=np.int16)
    else:
        signal_format_string = ', '.join(sorted(set(signal_formats)))
        raise NotImplementedError(f'{signal_format_string} not implemented')

    output_fields = dict()
    output_fields['sig_ID'] = helper_code.get_signal_files_from_header(header)[0]
    output_fields['sig_name'] = signal_names
    output_fields['fs'] = sampling_frequency
    output_fields['units'] = signal_units
    output_fields['comments'] = ""
    record = wfdb.Record(fs=sampling_frequency, units=signal_units, sig_name=signal_names,
                d_signal=signal, fmt=signal_formats, adc_gain=adc_gains, baseline=baselines, comments=comments,
                )
    
    return signal, output_fields, record


def plot_signal_reconstruction(label_signal, output_signal, output_fields, label_fields, trace, filename="", output_folder=""):
    # trace.save("trace.png")

    dsa = output_signal.to_numpy()
    fig, axs = plt.subplots(dsa.shape[1], 1, figsize=(10, 10))
    for i in range(dsa.shape[1]):
        axs[i].plot(dsa[:, i])
        axs[i].plot(label_signal[:, i])
    
    # combine this plot with original image and trace

    # save everything in one image
    # plt.savefig(os.path.join(output_folder, filename))
    plt.close()


def main(data_folder, records, verbose):
    # TODO: pass in list of record names
    if verbose:
        print('Extracting features and labels from the data...')
        t1 = time.time()

    num_records = len(records)
    features = list()

    for i in tqdm(range(num_records), disable=~verbose):  
        record = os.path.join(data_folder, records[i])

        # get ground truth signal and metadata
        header_file = helper_code.get_header_file(record)
        header = helper_code.load_text(header_file)

        num_samples = helper_code.get_num_samples(header)
        num_signals = helper_code.get_num_signals(header)

        # get filenames/paths of all records to be reconstructed
        image_files = team_helper_code.load_image_paths(record)
        image_file = image_files[0]
        if len(image_files) > 1:
            if verbose:
                print(f"Multiple images found, using image at {image_file}.")
        
        # run the digitization model
        trace, signal, gridsize = run_digitization_model(image_file, verbose)

        # compute SNR vs ground truth

        # add SNR, image metadata, label, and image filename to dataframe

        # plot original image, cleaned image with trace, ground truth signal, and reconstructed signal

        # save plots to output folder
    
    # save dataframe to output folder


    if verbose:
        t2 = time.time()
        print(f'Done. Time to extract features: {t2 - t1:.2f} seconds.')
        
        pass


