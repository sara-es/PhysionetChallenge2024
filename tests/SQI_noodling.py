# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:11:53 2024

@author: hssdwo
"""
# test script for signal SQI. take the output in pixel format.
# Filter the signal by setting each column to the nearest ROI.
# Find baseline for each signal
# If the signal has *extended* continuous period away from baseline, the SQI has failed
import numpy as np

# rois = 
SQI = True
thresh_lens = 5 # set consistency (in pixels) between rows. i.e. reject if row length are different

sig_len = []
for signal in enumerate(signals):
    sig_len.append(len(signal))
    
if (np.max(sig_len) - np.min(sig_len)) > thresh_lens:
    SQI = False

thresh_secs = 0.1 # set the maximum amount of time we allow a signal to be off baseline
thresh = len(signals[0])/(10/thresh_secs)

import numpy as np
for i, signal in enumerate(signals):
    filt_signal = []
    sig = []
    for j in range(len(signal)):
        #replace y value with nearest roi
        idx = np.where(np.abs(rois - signal[j][0]) == np.min(np.abs(rois - signal[j][0])))
        idx = idx[0][0]
        filt_signal.append(bool(rois[idx] - rois[i]))
        sig.append(signal[j][0])
    
    plt.figure()
    plt.plot(sig)

    # count the longest contiguous section of non-zeros
    section = 0
    this_section = 0
    #plt.figure()
    #plt.plot(filt_signal)
    for j in range(len(signal)):
        if filt_signal[j]:
            this_section = this_section + 1
        else:
            if this_section > section:
                section = this_section
            this_section = 0
            
            
    # if section is longer than threshold (some fraction of total length), then SQI fails
    if section > thresh:
            SQI = False