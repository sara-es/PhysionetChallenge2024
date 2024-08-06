# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:19:06 2024

Signal Quality Assessment at the end of digitization. For the challenge, it is better
to return zeroes if we think that our signal is a bit dodgy. The challenge metric (SNR)
is VERY harsh, so we should return zeroes if we think the lead has been lost for ANY
period of time

Input:
signals: output of ECG-miner reconstruction stage - a list of Point objects

Output:
    ret

@author: hssdwo
"""
import numpy as np

# TODO: convert miner list of Points back into a list of arrays.
def point_to_list(signals):
    signal_list = []
    for i in range(len(signals)):
        #signal_x = [p.x for p in signals[i]]
        signal_y = [p.y for p in signals[i]]
        
        signal_list.append(signal_y)
    return signal_list

#--------------------------------------------------------------------------------------------------------------------------------

# for the moment use signal_arr
signals = point_to_list(raw_s)

#rois = get_roi(image, n = 4):
    
roi_gap = np.mean(np.diff(rois)) # gap in pixels
thresh = roi_gap/ 5 #threshold to make decision. INCREASE to make it less sensitive
GOOD_SIGNAL = True

# 0. reject if there is much difference in length - this means that the line joining has failed for some reason
lengths = []
for signal in signals:
    lengths.append(len(signal))
if (max(lengths) - min(lengths)) > thresh:
    GOOD_SIGNAL = False
    #TODO return GOOD_SIGNAL


win_len = lengths[0] // 20 # assume that a row is always ten seconds, so win_len is roughly 0.5 secs
overlap = win_len // 5 # arbitrarily overlap by 1/5th of a window

for idx, signal in enumerate(signals):
    # 1. reject if one lead is identical to another for any portion of n-seconds
    
    
    
    # 2. reject if any lead is split over two baselines (i.e. in a given window, *most* of the points are a baseline away)
    
    # get windows:
    win_starts = []
    win_ends = []
    
    i = 0
    while i < (lengths[idx] - win_len):
        win_starts.append(i)
        win_ends.append(i + win_len)
        i = i+overlap
    
    # for each window, check if median is close to one of the neighbouring baselines
    this_baseline = rois[idx]
    next_baselines = [this_baseline-roi_gap, this_baseline+roi_gap]
    for i, win in enumerate(win_starts):
        window = signal[win_starts[i]:win_ends[i]]
        if any(abs(np.median(window) - next_baselines) < thresh):
            GOOD_SIGNAL = False
            break