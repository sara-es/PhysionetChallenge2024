# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:46:39 2024

@author: hssdwo
"""

#%%
import numpy as np
from scipy.signal import correlate
from digitization.ECGminer.assets.Lead import Lead


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


'''Detects cabrera layout - increase thresh to make decrease false positive rate at expense of true positive'''
def cabrera_detector(signal, NCOLS = 4, thresh = 500):
     diff_signals = np.diff(signal)
     ref = np.size(diff_signals,1)
     col_width = ref//NCOLS
 
     if NCOLS == 4:
         pos1 = diff_signals[0,1:col_width]
         pos2 = diff_signals[1,1:col_width]
         pos3 = diff_signals[2,1:col_width]
         pos4 = diff_signals[0,col_width:2*col_width]
         pos5 = diff_signals[1,col_width:2*col_width]
         pos6 = diff_signals[2,col_width:2*col_width]
     
         test = correlate(pos1, pos2, mode='same', method='auto')
         idx = np.where(abs(test)==max(abs(test)))[0]
         corr1 = test[idx][0]
         test = correlate(pos2, pos3, mode='same', method='auto')
         idx = np.where(abs(test)==max(abs(test)))[0]
         corr2 = test[idx][0]
         test = correlate(pos3, pos4, mode='same', method='auto')
         idx = np.where(abs(test)==max(abs(test)))[0]
         corr3 = test[idx][0]
         test = correlate(pos4, pos5, mode='same', method='auto')
         idx = np.where(abs(test)==max(abs(test)))[0]
         corr4 = test[idx][0]
         test = correlate(pos5, pos6, mode='same', method='auto')
         idx = np.where(abs(test)==max(abs(test)))[0]
         corr5 = test[idx][0]
     
         test = correlate(pos5, pos1, mode='same', method='auto')
         idx = np.where(abs(test)==max(abs(test)))[0]
         c_corr1 = test[idx][0]
         test = correlate(pos1, -pos4, mode='same', method='auto')
         idx = np.where(abs(test)==max(abs(test)))[0]
         c_corr2 = test[idx][0]
         test = correlate(-pos4, pos2, mode='same', method='auto')
         idx = np.where(abs(test)==max(abs(test)))[0]
         c_corr3 = test[idx][0]
         test = correlate(pos2, pos6, mode='same', method='auto')
         idx = np.where(abs(test)==max(abs(test)))[0]
         c_corr4 = test[idx][0]
         test = correlate(pos6, pos3, mode='same', method='auto')
         idx = np.where(abs(test)==max(abs(test)))[0]
         c_corr5 = test[idx][0]
     
         a = np.std([corr1, corr2, corr3, corr4, corr5])
         b = np.std([c_corr1, c_corr2, c_corr3, c_corr4, c_corr5])

     if (a + thresh) > b:
         return False
     else:
         return True
     

def return_formats(is_cabrera):
    if is_cabrera:
        format_3x4 = [
            [Lead.aVL, Lead.II, Lead.V1, Lead.V4],
            [Lead.I, Lead.aVF, Lead.V2, Lead.V5],
            [Lead.aVR, Lead.III, Lead.V3, Lead.V6],
        ]

        format_6x2 = [
            [Lead.aVL, Lead.V1],
            [Lead.I, Lead.V2],
            [Lead.aVR, Lead.V3],
            [Lead.II, Lead.V4],
            [Lead.aVF, Lead.V5],
            [Lead.III, Lead.V6],
        ]
    else:
        format_3x4 = [
            [Lead.I, Lead.aVR, Lead.V1, Lead.V4],
            [Lead.II, Lead.aVL, Lead.V2, Lead.V5],
            [Lead.III, Lead.aVF, Lead.V3, Lead.V6],
        ]
        
        format_6x2 = [
            [Lead.I, Lead.V1],
            [Lead.II, Lead.V2],
            [Lead.III, Lead.V3],
            [Lead.aVR, Lead.V4],
            [Lead.aVL, Lead.V5],
            [Lead.aVF, Lead.V6],
        ] 
    return format_3x4, format_6x2


def detect_rhythm_strip(signal_arr, is_cabrera, THRESH = 2):
    """
    This takes in a N-row array corresponding to the ECGs found by the line tracing algorithm.

    Parameters
        signal_arr: np.array of interpolated signals: correct len but still in pixel y-coords
        is_cabrera: bool, True if the ECG is in Cabrera format, False if in standard format
        THRESH: int, the maximum average difference in pixels between the rows to be accepted 
            as similar enough to be the rhythm strip
    """
    format_3x4, format_6x2 = return_formats(is_cabrera)
    NROWS = signal_arr.shape[0]

    #2.for each signal, get its np.diff - this converts from pixel coords to relative coords
    diff_signals = np.diff(signal_arr)
    rhythm = []
    fails = []
    NUM_SHORTROWS = 3
    
    if NROWS > NUM_SHORTROWS: # physionet assumption - everything is in 3x4 format
        NCOLS = 4
        test_grid = np.zeros([NUM_SHORTROWS,NCOLS])
        main_ecg = diff_signals[0:NUM_SHORTROWS,:]
        for i in range(NUM_SHORTROWS, NROWS):   
            ref = diff_signals[i,:]
            col_width = len(ref)//NCOLS
            
            # compare the ith row to the top 3 rows - this currently assumes they are all the
            # same length.
            test = abs(main_ecg - ref)
            
            for j in range(NUM_SHORTROWS):
                for k in range(NCOLS):
                    start = col_width*k
                    fin = col_width*(k+1)-1
                    test_grid[j,k] = sum(test[j][start:fin]) / col_width
            
            # find the minimum value of test_grid. If minimum value is less than threshold, 
            # report j,k. Otherwise, use defaults rhythm values
            if (np.min(test_grid) < THRESH): # i.e. on average, maximum of 1 pixel difference
                x,y = np.where(test_grid == np.min(test_grid))
                rhythm.append(format_3x4[x[0]][y[0]]) # get corresponding label
            else:
                #record offset idx of any fails
                rhythm.append('FAIL')
                fails.append(i-NUM_SHORTROWS)
            
        # Hacky code ahead - 
        if NROWS == 4:
            default_rhythm = [Lead.II]
        elif NROWS == 5:
            default_rhythm = [Lead.II, Lead.V5]
        elif NROWS == 6:
            default_rhythm = [Lead.V1, Lead.II, Lead.V5]
        else:
            # this should basically never happen
            default_rhythm = [Lead.V1, Lead.II, Lead.V5, Lead.V1, Lead.II, Lead.V5]
        # if we didn't find all of the rhythm strips successfully, default any fails to the most 
        # likely strip
        for j in fails:
            rhythm[j] = default_rhythm[j]
                
        # n.b. this row shouldn't be necessary at all, but it's a failsafe in case the logic above
        # has messed up in an unexpected way for the special case of 4 rows, and the auto finder 
        # fails - assume that the rhythm strip is lead II
        if (NROWS == 4) & (len(fails) > 0):
            rhythm = [Lead.II]
    
    return rhythm

''' The Physionet Challenge 2024 uses ONLY 3x4 format. The function below can replace any calls 
to detect_rhythm_strip and should automatically detect whether the ECG is in 3x4, 6x2 or 12x1 
layout (with or without multiple rhythm strips)
'''
def detect_rhythm_strip_general(signal_arr, is_cabrera, THRESH = 1):
    NROWS = np.shape(signal_arr)[0]
    
    format_3x4, format_6x2 = return_formats(is_cabrera)
    
    #2.for each signal, get its np.diff - this converts from pixel coordinates to 
    # relative coordinates
    diff_signals = np.diff(signal_arr)
    rhythm = []

    if NROWS ==12:# definititely 12x1
        layout = (12,1)  
        
    elif NROWS <= 6: # if there are rhythm strips, then we assume that the layout is in 3x4 format
        layout = (3,4)    
        NCOLS = 4
        test_grid = np.zeros([3,NCOLS])
        main_ecg = diff_signals[0:3,:]
        for i in range(3, NROWS):   
            ref = diff_signals[i,:]
            col_width = len(ref)//NCOLS
            
            # compare the ith row to the top 3 rows - this currently assumes they are all the
            # same length.
            test = abs(main_ecg - ref)
            
            for j in range(3):
                for k in range(NCOLS):
                    start = col_width*k
                    fin = col_width*(k+1)-1
                    test_grid[j,k] = sum(test[j][start:fin]) / col_width
            
            # find the minimum value of test_grid. If minimum value is less than threshold, 
            # report j,k.
            if (np.min(test_grid) < THRESH): # i.e. on average, maximum of 1 pixel difference
                x,y = np.where(test_grid == np.min(test_grid))
                rhythm.append(format_3x4[x[0]][y[0]]) # get corresponding label
    
        #for the special case of 4 rows, and the auto finder fails - assume that the rhythm strip
        # is lead II
        if (NROWS == 4) & (len(rhythm) == 0):
            rhythm.append('Lead.II') 
         
        if (NROWS == 6) & (len(rhythm) == 0):
            layout = (6,2)
    
    elif NROWS <= 11:
        layout = (6,2)
        NCOLS = 2
        test_grid = np.zeros([6,NCOLS])
        main_ecg = diff_signals[0:6,:]
        for i in range(6, NROWS):
            ref = diff_signals[i,:]
            col_width = len(ref)//NCOLS
            
            # compare the ith row to the top 3 rows - this currently assumes they are all the same
            # length.
            test = abs(main_ecg - ref)
            
            for j in range(6):
                for k in range(NCOLS):
                    start = col_width*k
                    fin = col_width*(k+1)-1
                    test_grid[j,k] = sum(test[j][start:fin]) / col_width
            
            # find the minimum value of test_grid. If minimum value is less than threshold, 
            # report j,k.
            if (np.min(test_grid) < THRESH): # i.e. on average, maximum of 1 pixel difference
                x,y = np.where(test_grid == np.min(test_grid))
                rhythm.append(format_6x2[x[0]][y[0]]) # get corresponding label
    return layout, rhythm
