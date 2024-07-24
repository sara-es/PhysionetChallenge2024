# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:46:39 2024

@author: hssdwo
"""

#%%
import numpy as np
from digitization.ECGminer.assets.Lead import Lead

#%%
def point_to_matrix(signals):
    #convert list of points into matrix of signals - only contains points that overlap in all raws
    max_x = 10000
    rows = len(signals)
    for i in range(rows):
        row_max = signals[i][-1].x
        if row_max < max_x:
            max_x = row_max
    
    signal_arr = np.zeros([4, (max_x + 1)])
    
    for i in range(rows):
        signal_x = [p.x for p in signals[i]]
        signal_y = [p.y for p in signals[i]]
        
        
        for idx, j in enumerate(signal_x):
            signal_arr[i, j] = signal_y[idx]
    
    # strip out any columns that contains zeros
    idx = np.where(np.sum(signal_arr,0) == 0)[0]
    signal_arr = np.delete(signal_arr, idx, 1)
    
    return signal_arr

#%%
"""
This takes in a N-row array corresponding to the ECGs found by the line tracing algorithm.
Input: list of ECG signals in the ecg-miner Point object format
Output: list of strings (rhythm strip lead name) in order from top to bottom of image.
The user parameter THRESH is the average per-column difference allowed between the rows to be accepted as the rhythm
"""

def detect_rhythm_strip(signal_arr, THRESH = 1):
    """
    Parameters
        signal_arr: np.array
    """
    # TODO: move this constant to somewhere sensible if needed
    format_3x4 = [
        [Lead.I, Lead.aVR,Lead.V1, Lead.V4],
        [Lead.II, Lead.aVL, Lead.V2, Lead.V5],
        [Lead.III, Lead.aVF, Lead.V3, Lead.V6],
    ]

    # format_6x2 = [
    #     ['Lead.I', 'Lead.V1'],
    #     ['Lead.II', 'Lead.V2'],
    #     ['Lead.III', 'Lead.V3'],
    #     ['Lead.aVR', 'Lead.V4'],
    #     ['Lead.aVL', 'Lead.V5'],
    #     ['Lead.aVF', 'Lead.V6'],
    # ]

    # format_12x1 = [
    #     'Lead.I',
    #     'Lead.II',
    #     'Lead.III',
    #     'Lead.aVR',
    #     'Lead.aVL',
    #     'Lead.aVF',
    #     'Lead.V1',
    #     'Lead.V2',
    #     'Lead.V3',
    #     'Lead.V4',
    #     'Lead.V5',
    #     'Lead.V6',
    # ]

    n_rows = signal_arr.shape[0]
    
    #2.for each signal, get its np.diff - this converts from pixel coordinates to relative coordinates
    diff_signals = np.diff(signal_arr)
    reference = []

    if n_rows <=6: # if there are reference pulses, then we assume that the layout is in 3x4 format
        NCOLS = 4
        test_grid = np.zeros([3,NCOLS])
        for i in range(3, n_rows):
            main_ecg = diff_signals[0:3,:]
            ref = diff_signals[i,:]
            col_width = len(ref)//NCOLS
            
            # compare the ith row to the top 3 rows - this currently assumes they are all the same length.
            test = abs(main_ecg - ref)
            
            for j in range(3):
                for k in range(NCOLS):
                    start = col_width*k
                    fin = col_width*(k+1)-1
                    test_grid[j,k] = sum(test[j][start:fin]) / col_width
            
            # find the minimum value of test_grid. If minimum value is less than threshold, report j,k.
            if (np.min(test_grid) < THRESH): # i.e. on average, maximum of 1 pixel difference
                x,y = np.where(test_grid == np.min(test_grid))
                reference.append(format_3x4[x[0]][y[0]]) # get corresponding label
    
    #for the special case of 4 rows, and the auto finder fails - assume that the rhythm strip is lead II
    if n_rows == 4 & len(reference) == 0:
        reference.append(Lead.II)
                
    ## TODO  - extend to work with 6x2 format. Code below might work, but not tested
    
    # elif NROWS <=11: # if there are reference pulses, then we assume that the layout is in 6x2 format
    #     NCOLS = 2
    #     for i in range(6, NROWS):
    #         main_ecg = diff_signals[0:2]
    #         ref = diff_signals[i]
        
    #         # compare the ith row to the top 3 rows
    #         test = main_ecg - ref
        
    #         for j in range(2):
    #             for k in range(NCOLS):
    #             test_grid[j,k] = sum(test[j][0:100])
        
    #         # find the minimum value of test_grid. If minimum value is less than threshold, report j,k.
        
    #         # convert j,k to lead name and store in rhythm
    
    # To differentiate between standard and cabrera, compare similarity of lead III and aVF
    
    return reference