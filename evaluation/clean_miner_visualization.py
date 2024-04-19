"""
A scipt to visualize the results of the digitization model pipeline on the Challenge data.

Example usage follows the convention of the other Challenge scripts:
python clean_miner_visualization.py -i [input_folder] -o [output_folder] -v

Note that the input folder must have image files as well as ground truth signal files.
"""


import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import traceback, argparse, time
import wfdb
from tqdm import tqdm
from PIL import Image

import team_code, helper_code
from reconstruction import image_cleaning
from utils import team_helper_code


def run_digitization_model(image_file, sig_len, verbose, allow_failures=False):
    # clean and rotate the image
    cleaned_image, gridsize = image_cleaning.clean_image(image_file)   

    # digitize with ECG-miner
    try:
        signal, trace = image_cleaning.digitize_image(cleaned_image, gridsize, sig_len)
        signal = np.nan_to_num(signal)
    except Exception as e: 
        if allow_failures:
            signal = None
            if verbose:
                print(f"Error digitizing image {image_file}: {e}")
                print(traceback.format_exc())
        else: raise e

    return trace, signal, gridsize


def match_signal_lengths(output_signal, output_fields, label_signal, label_fields):
    # make sure channel orders match, and trim/pad output signal to match label signal length
    if label_signal is not None:
        label_channels = label_fields['sig_name']
        label_num_samples = label_signal.shape[0]
        label_sampling_frequency = label_fields['fs']
        label_units = label_fields['units']

        if output_signal is not None:
            output_channels = output_fields['sig_name']
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
    
    return output_signal, output_fields, label_signal, label_fields


def single_signal_snr(output_signal, output_fields, label_signal, label_fields, extra_scores=False):
    # lifted from evaluate_model.py with minor edits
    snr = dict()
    snr_median = dict()
    ks_metric = dict()
    asci_metric = dict()
    weighted_absolute_difference_metric = dict()

    record = output_fields['sig_ID']
    label_channels = label_fields['sig_name']
    label_num_channels = label_signal.shape[1]
    label_sampling_frequency = label_fields['fs']

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
    signal = np.asarray(signal, dtype=float)/1000.0
    
    return signal, output_fields, record




def plot_signal_reconstruction(label_signal, output_signal, output_fields, mean_snr, trace, image_file, output_folder="", gridsize=None):
    description_string = f"""{output_fields['sig_ID']} ({output_fields['fs']} Hz)
    Reconstruction SNR: {mean_snr:.2f}
    Gridsize: {gridsize}"""

    # # plot original image, cleaned image with trace, ground truth signal, and reconstructed signal  
    mosaic = plt.figure(layout="tight", figsize=(18, 13))
    axd = mosaic.subplot_mosaic([
                                    ['original_image', 'ecg_plots'],
                                    ['trace', 'ecg_plots']
                                ])
    # plot original image
    with Image.open(image_file) as img:
        axd['original_image'].axis('off')
        axd['original_image'].imshow(img, cmap='gray')
        
    # plot trace
    axd['trace'].xaxis.set_visible(False)
    axd['trace'].yaxis.set_visible(False)
    axd['trace'].imshow(trace.data, cmap='gray')    

    # plot ground truth and reconstructed signal
    fig, axs = plt.subplots(output_signal.shape[1], 1, figsize=(9, 12.5), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle(description_string)
    # fig.text(0.5, 0.95, description_string, ha='center')
    for i in range(output_signal.shape[1]):
        axs[i].plot(output_signal[:, i])
        axs[i].plot(label_signal[:, i])
    fig.canvas.draw()
    axd['ecg_plots'].axis('off')
    axd['ecg_plots'].imshow(fig.canvas.renderer.buffer_rgba())
    # save everything in one image
    filename = f"{output_fields['sig_ID']}_reconstruction.png"
    plt.figure(mosaic)
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()


def main(data_folder, output_folder, verbose):
    # Find data files.
    records = helper_code.find_records(data_folder)
    if len(records) == 0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    for i in tqdm(range(len(records)), disable=~verbose):  
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
        frequency = helper_code.get_sampling_frequency(header)
        longest_signal_length = frequency*num_samples
        trace, signal, gridsize = run_digitization_model(image_file, longest_signal_length, verbose=True)
        breakpoint()
        
        # get digitization output
        # signal = np.asarray(signal, dtype=np.int16)
        # run_model.py just rewrites header file to output folder here, so we can skip that step
        # also saves the signal as a wfdb signal
        output_signal, output_fields, wfdb_signal = format_wfdb_signal(header, signal) # output_record is the filepath the output signal will be saved to
        breakpoint()
        # get ground truth signal
        label_signal, label_fields = helper_code.load_signal(record)
        # match signal lengths: make sure channel orders match and trim output signal to match label signal length
        output_signal, output_fields, label_signal, label_fields = match_signal_lengths(output_signal, output_fields, label_signal, label_fields)
        # breakpoint()
        # compute SNR vs ground truth
        mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, mean_weighted_absolute_difference_metric = single_signal_snr(output_signal, output_fields, label_signal, label_fields, extra_scores=True)
        
        # save dataframe to output folder
        #TODO

        # plot signal reconstruction
        plot_signal_reconstruction(label_signal, output_signal, output_fields, mean_snr, trace, image_file, output_folder, gridsize=gridsize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the results of the digitization model pipeline on the Challenge data.')
    parser.add_argument('-i', '--data_folder', type=str, help='The folder containing the Challenge images.')
    parser.add_argument('-o', '--output_folder', type=str, help='The folder to save the output visualization.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print progress messages.')
    args = parser.parse_args()

    main(args.data_folder, args.output_folder, args.verbose)


