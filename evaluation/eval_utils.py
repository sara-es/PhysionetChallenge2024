"""
Functions that I call in more than one evaluation script.

A lot of the code is adapted from an old version of helper_code.py, which has since been updated
by the Challenge team, so use at your own risk.
"""

import os, sys
os.path.join(sys.path[0], '..')

import numpy as np

import helper_code


def match_signal_lengths(output_signal, output_fields, label_signal, label_fields):
    # make sure channel orders match, and trim/pad output signal to match label signal length
    if label_signal is not None:
        label_channels = label_fields['sig_name']
        label_n_channels = label_fields['n_sig']
        label_num_samples = label_signal.shape[0]
        label_sampling_frequency = label_fields['fs']
        label_units = label_fields['units']

        if output_fields is not None:
            output_channels = output_fields['sig_name']
            output_sampling_frequency = output_fields['fs']
            output_units = output_fields['units']
            output_n_channels = output_fields['n_sig']

            # Check that the label and output signals match as expected.
            assert(label_sampling_frequency == output_sampling_frequency)
            assert(label_units == output_units)

            # Reorder the channels in the output signal to match the channels in the label signal.
            output_signal = helper_code.reorder_signal(output_signal, output_channels, 
                                                       label_channels)

        else:
            output_signal = np.nan*np.ones(np.shape(label_signal))
    
    return output_signal, output_fields, label_signal, label_fields


def trim_label_signal(input_signal, signal_names, num_samples_trimmed, 
                      layout=(3,4),  rhythm=['II']):
    """
    Trim the label signal to show only what would have been plotted in a generated image,
    replacing the rest with zeros.

    params:
        label_signal: numpy array shape (nsamples, nleads), signal to be trimmed
        signal_names: list of strings, names of the leads in the signal
        num_samples_trimmed: int, desired number of samples of all but rhythm
        layout: tuple (nrows, ncols), not including rhythm
        rhythm: list of strings, names of the leads to keep at full length
    returns:
        trimmed_label_signal: numpy array, trimmed signal    
    """
    input_signal = np.asarray(input_signal)
    trimmed_signal = np.zeros_like(input_signal)
    trimmed_signal[:] = np.nan # set to nan to account for new scoring metric

    for i, lead in enumerate(signal_names):
        if lead in rhythm:
            trimmed_signal[:, i] = input_signal[:, i]
        else:
            start = int(num_samples_trimmed*(i // layout[0]))
            end = int(num_samples_trimmed*((i // layout[0]) + 1))
            trimmed_signal[start:end, i] = input_signal[start:end, i]
    return trimmed_signal


def single_signal_snr(output_signal, output_fields, label_signal, label_fields, record, 
                      extra_scores=False, no_shift=False):
    # lifted from evaluate_model.py with minor edits
    snr = dict()
    snr_median = dict()
    ks_metric = dict()
    asci_metric = dict()
    weighted_absolute_difference_metric = dict()

    channels = label_fields['sig_name']
    sampling_frequency = label_fields['fs']

     # Set limits on how far the signal can be shifted, and the number of quantization levels when shifting the signals.
    max_hz_shift = np.round(0.5*sampling_frequency)
    max_vt_shift = 1.0
    num_quant_levels = 2**8

    # Shift the digitied signals to better align with the reference signals.
    signal_ref_collection = list()
    signal_est_collection = list()
    for j, channel in enumerate(channels):
        signal_ref_collection.append(label_signal[:, j])
        # Align the signals.
        if not no_shift:
            signal_shifted, shift_hz, shift_vt = helper_code.align_signals(label_signal[:, j], output_signal[:, j], num_quant_levels=num_quant_levels)
            if abs(shift_hz) <= max_hz_shift and abs(shift_vt) <= max_vt_shift:
                signal_est_collection.append(signal_shifted)
            else:
                signal_est_collection.append(output_signal[:, j])
        else:
            signal_est_collection.append(output_signal[:, j])

    # Compute the signal reconstruction metrics.
    for j, channel in enumerate(channels):
        value, p_signal, p_noise = helper_code.compute_snr(signal_ref_collection[j], signal_est_collection[j])
        snr[(record, channel)] = value

        if extra_scores:
            value = helper_code.compute_snr(signal_ref_collection[j], signal_est_collection[j], noise_median=True)
            snr_median[(record, channel)] = value

            value = helper_code.compute_ks_metric(signal_ref_collection[j], signal_est_collection[j])
            ks_metric[(record, channel)] = value

            value = helper_code.compute_asci_metric(signal_ref_collection[j], signal_est_collection[j])
            asci_metric[(record, channel)] = value

            value = helper_code.compute_weighted_absolute_difference(signal_ref_collection[j], signal_est_collection[j], sampling_frequency)
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

        weighted_absolute_difference_metric = np.array(
            list(weighted_absolute_difference_metric.values()))
        if not np.all(np.isnan(weighted_absolute_difference_metric)):
            mean_weighted_absolute_difference_metric = np.nanmean(
                weighted_absolute_difference_metric)
        else:
            mean_weighted_absolute_difference_metric = float('nan')
    else:
        mean_snr_median = float('nan')
        mean_ks_metric = float('nan')
        mean_asci_metric = float('nan')
        mean_weighted_absolute_difference_metric = float('nan')
    
    return mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, \
        mean_weighted_absolute_difference_metric