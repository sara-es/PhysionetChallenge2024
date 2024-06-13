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


def single_signal_snr(output_signal, output_fields, label_signal, label_fields, record, extra_scores=False):
    # lifted from evaluate_model.py with minor edits
    snr = dict()
    snr_median = dict()
    ks_metric = dict()
    asci_metric = dict()
    weighted_absolute_difference_metric = dict()

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