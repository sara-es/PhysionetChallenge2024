# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:18:52 2024

@author: hssdwo
"""


import pickle
import numpy as np

def is_pulse(signal, baseline, min_width = 20, min_height = 25, tol = 3):
    for i in range(len(signal)-min_width):
        segment = signal[i:i+(min_width)]
        seg_range = max(segment)-min(segment)
        seg_median = np.median(segment)
        offset = seg_median - baseline
        if (seg_range < tol) & (offset > min_height):
            return True
    return False

def row_pulse_pos(signal, scan_perc = 0.1):
    baseline = np.median(signal)
    scan_length = round(len(signal) * scan_perc) # look in the first and last 10% of the signal
    INI, MID, END, NONE = (0, 1, 2, 3)
    
    # get left hand segment
    left_segment = signal[0:scan_length]
    right_segment = signal[-scan_length:]
    is_left = is_pulse(left_segment, baseline)
    is_right = is_pulse(right_segment, baseline)

    if is_left & is_right:
        return NONE
    elif is_left:
        return INI
    elif is_right:
        return END

def pulse_pos(data, scan_perc = 0.1):
    INI, MID, END, NONE = (0, 1, 2, 3)
    pulses = np.empty(0)
    for i in range(data.shape[0]): #for each row
        data_row = -data[i,:]
        pos = row_pulse_pos(data_row)
        print(pos)
        pulses = np.append(pulses,pos)
    
    if sum(pulses == INI)>=2:
        return INI
    elif sum(pulses == END)>=2:
        return END
    else:
        return NONE

    

# data_directory = 'C:/Users/hssdwo/Downloads/lr_gt_refpulse/01033_lr_raw_signal.pkl'
# with open(data_directory, 'rb') as f:
#     data = pickle.load(f)
    
# pulses = pulse_pos(data)






