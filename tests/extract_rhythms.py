# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:46:39 2024

@author: hssdwo
"""

# input - list of points in a row
#1.convert the points into a signal - this is probably already done in ecg-miner

#%%
import numpy as np
import pickle
from scipy.signal import find_peaks
from digitization.ECGminer.assets.Point import Point

format_3x4 = [
    ['Lead.I', 'Lead.aVR','Lead.V1', 'Lead.V4'],
    ['Lead.II', 'Lead.aVL', 'Lead.V2', 'Lead.V5'],
    ['Lead.III', 'Lead.aVF', 'Lead.V3', 'Lead.V6'],
]

format_3x4_cab = [
    ['Lead.aVL', 'Lead.II','Lead.V1', 'Lead.V4'],
    ['Lead.I', 'Lead.aVF', 'Lead.V2', 'Lead.V5'],
    ['Lead.-aVR', 'Lead.III', 'Lead.V3', 'Lead.V6'],
]

format_6x2 = [
    ['Lead.I', 'Lead.V1'],
    ['Lead.II', 'Lead.V2'],
    ['Lead.III', 'Lead.V3'],
    ['Lead.aVR', 'Lead.V4'],
    ['Lead.aVL', 'Lead.V5'],
    ['Lead.aVF', 'Lead.V6'],
]


format_6x2_cab = [
    ['Lead.aVL', 'Lead.V1'],
    ['Lead.I', 'Lead.V2'],
    ['Lead.-aVR', 'Lead.V3'],
    ['Lead.II', 'Lead.V4'],
    ['Lead.aVF', 'Lead.V5'],
    ['Lead.III', 'Lead.V6'],
]

# TODO: fix to automatically detect more rows

def get_roi(image, n = 4):
    """
    Get the coordinates of the ROI of the ECG image.

    Args:
        ecg (Image): ECG image from which to extract the ROI.

    Raises:
        DigitizationError: The indicated number of ROI could not be detected.

    Returns:
        Iterable[int]: List of row coordinates of the ROI.
    """
    height = image.shape[0]
    width = image.shape[1]
    WINDOW = 10
    SHIFT = (WINDOW - 1) // 2
    stds = np.zeros(height)
    for i in range(height - WINDOW + 1):
        x0, x1 = (0, width)
        y0, y1 = (i, i + WINDOW - 1)
        std = image[y0:y1, x0:x1].reshape(-1).std()
        stds[i + SHIFT] = std
    # Find peaks
    min_distance = int(height * 0.1)
    peaks, _ = find_peaks(stds, distance=min_distance)
    rois = sorted(peaks, key=lambda x: stds[x], reverse=True)
    if len(rois) < n:
        print("The indicated number of rois could not be detected.")
    rois = rois[0: n]
    rois = sorted(rois)
    return rois

"""Find the start position of the ECG signal for each row.
N.B. This will fail if signal from an adjacent row gets close to the baseline for this row.
"""
def find_start_coords(image, rois, row):
    # assume that the starting point is within 0.75 * gap between lines. This may not be true if signal starts on a particularly high/low QRS complex
    roi_gap = int(np.max(np.diff(rois)) *0.75)
    height = image.shape[0]
    
    top = row - roi_gap
    if top < 0:
        top = 0
    bottom = row + roi_gap
    if bottom > height:
        bottom = height
    
    # find the starting and end column - columns with black pixels within the active region
    colsums = np.min(image[top:bottom,:],0)
    idxs = np.where(colsums == 0)[0]
    startcol = idxs[0]
    endcol = idxs[-1]
    
    # find the start row - the black pixel that is nearest to the baseline. This also doesn't work if adjacent row bleeds into this row
    rowvals = image[:,startcol]
    idxs = np.where(rowvals == 0)[0]
    idx = np.argmin(abs(idxs - row))
    startrow = idxs[idx]
    return startrow, startcol, endcol

"""split the ECG image into its constituent rows. The logic is as follows:
   - search up and down to find the ends of the lines
   - from the line ends, find nearest point in the next column (assume there is only one, as unet is good)
   - repeat until the end

thresh is a constant that defines the maximum allowable jump between columns, in pixels. If we *fully* trust gridsize, then can make this a percentage of gridsize instead
"""
def extract_rows(image, thresh = 50, plot_output = True):
    test_im = abs(image -1)*255 # invert colours
    
    rois = get_roi(test_im)
    s = []
    
    for row in rois:
        x, y, endcol = find_start_coords(test_im, rois, row)
        signal = []
        signal_col = []
        signal_col.append([x,y])
        
        while y < endcol:
            # search up
            x_idx = x-1
            while test_im[x_idx, y] == 0:
                signal_col.append([x_idx,y])
                x_idx = x_idx-1
            # search down
            x_idx = x+1
            while test_im[x_idx, y] == 0:
                signal_col.append([x_idx,y])
                x_idx = x_idx+1
        
            signal_col = sorted(signal_col, key=lambda x: x[0], reverse=False)
            signal.append(signal_col)
            
            # go to next column and find nearest black pixels
            top_line = signal_col[0][0] # y-coordinate of top of line in previous column 
            bottom_line = signal_col[-1][0] # y-coordinate of bottom of line in previous column 
            y = y+1
            match = 0
            while match == 0:
                candidates = np.where(test_im[:,y] == 0)[0]
                dists = np.concatenate((candidates - top_line, candidates - bottom_line))
                candidates = np.concatenate((candidates, candidates))
                if np.min(abs(dists))<thresh:                    
                    #bit of voodoo here to try to make sure that we favour returning towards baseline when there is a choice
                    dist_idx = np.where(abs(dists) == min(abs(dists)))[0]
                    if len(dist_idx)>1:
                        #if there is more than one minimum, select the one that takes us closer to the baseline
                        x = candidates[dist_idx]
                        i = np.argmin(abs(x-row))
                        x = x[i]
                    else:
                        x = candidates[dist_idx][0]     
                    signal_col = []
                    signal_col.append([x,y])
                    match = 1
                elif y == endcol:
                    match = 1
                else: #skip a column and try again
                    y = y+1
                    
        #sanitise the list
        signal_flat = [x for xs in signal for x in xs]
        s.append(signal_flat)
    return s, rois
        
"""convert ecg sparse matrices into ecg miner format"""
def signal_to_miner(signals, rois):
    all_sigs = []
    for i, signal in enumerate(signals):
        start = signal[0][1]
        fin = signal[-1][1]
        raw_s = [None] * (fin-start + 1)
        sig = np.array(signal)
        roi = rois[i]
        for j, col in enumerate(range(start, fin+1)):
            idxs = np.where(sig[:,1]==col)[0]
            yvals = sig[idxs][:,0]
            if len(yvals)>0:
                max_idx = np.argmax(np.abs(yvals - roi))
                raw_s[j] = Point(col, yvals[max_idx])
                y = yvals[max_idx]
            else:
                # if we skipped a column, then use y val from previous column
                raw_s[j] = Point(col, y)
        all_sigs.append(raw_s)
    return all_sigs

#%%
file = "C:/Users/hssdwo/Downloads/images.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)

test_im = data[5]
signals, rois = extract_rows(test_im, thresh = 50, plot_output = False)
raw_s = signal_to_miner(signals, rois)

#%%
def point_to_matrix(signals):
    #convert list of points into matrix of signals - only contains points that overlap in all raws
    max_x = 10000
    rows = len(signals)
    for i in range(rows):
        row_max = signals[i][-1].x
        if row_max < max_x:
            max_x = row_max
    
    signal_arr = np.zeros([rows, (max_x + 2)]) #TODO: check that this works for all cases. adding 1 broke...
    
    for i in range(rows):
        signal_x = [p.x for p in signals[i]]
        signal_y = [p.y for p in signals[i]]
        
        
        for idx, j in enumerate(signal_x):
            signal_arr[i, j] = signal_y[idx]
    
    # strip out any columns that contains zeros
    has_num =  np.sum((signal_arr == 0) == False,0)
    idx = np.where(has_num != rows)[0]
    signal_arr = np.delete(signal_arr, idx, 1)
    
    return signal_arr

#%%
"""
This takes in a N-row array corresponding to the ECGs found by the line tracing algorithm.
Input:
    - signal_arr: list of ECG signals in the ecg-miner Point object format
    - THRESH: The user parameter THRESH is the average per-column difference allowed between the rows to be accepted as the rhythm
    - layout: normal or cabrera, derived from cabrera_detection function
Output: list of strings (rhythm strip lead name) in order from top to bottom of image.


TODO: lots of repeating code in the IF - refactor if there is time
"""

def detect_rhythm_strip(signal_arr, THRESH = 1, layout = 'normal'):
    NROWS = len(signals)

    #2.for each signal, get its np.diff - this converts from pixel coordinates to relative coordinates
    diff_signals = np.diff(signal_arr)
    reference = []

    if NROWS <=6: # if there are reference pulses, then we assume that the layout is in 3x4 format
        #assume 3x4 format:
        ECG_ROWS = 3
        ECG_COLS = 4
        test_grid = np.zeros([ECG_ROWS,ECG_COLS])
        for i in range(ECG_ROWS, NROWS):
            main_ecg = diff_signals[0:ECG_ROWS,:] # 12 leads are contained in the top 3 rows
            ref = diff_signals[i,:]
            col_width = len(ref)//ECG_COLS
            
            # compare the ith row to the top 3 rows - this currently assumes they are all the same length.
            test = abs(main_ecg - ref)
            
            for j in range(ECG_ROWS):
                for k in range(ECG_COLS):
                    start = col_width*k
                    fin = col_width*(k+1)-1
                    test_grid[j,k] = sum(test[j][start:fin]) / col_width
            
            # find the minimum value of test_grid. If minimum value is less than threshold, report j,k.
            if (np.min(test_grid) < THRESH): # i.e. on average, maximum of 1 pixel difference
                x,y = np.where(test_grid == np.min(test_grid))
                if layout == 'cabrera':
                    reference.append(format_3x4_cab[x[0]][y[0]]) # get corresponding label
                else:    
                    reference.append(format_3x4[x[0]][y[0]]) # get corresponding label
                    
    
        #for the special case of 4 rows, and the auto finder fails - assume that the rhythm strip is lead II
        if NROWS == 4 & len(reference) == 0:
            reference.append('Lead.II')
                
    ## TODO  - extend to work with 6x2 format. Code below might work, but not tested
    elif NROWS <=11: # if there are reference pulses, then we assume that the layout is in 6x2 format
        ECG_ROWS = 6
        ECG_COLS = 2
        for i in range(ECG_ROWS, NROWS):
            main_ecg = diff_signals[0:ECG_ROWS,:] # 12 leads are contained in the top 6 rows
            ref = diff_signals[i,:]
            col_width = len(ref)//ECG_COLS
            
            # compare the ith row to the top 6 rows
            test = main_ecg - ref
        
            for j in range(ECG_ROWS):
                for k in range(ECG_COLS):
                    start = col_width*k
                    fin = col_width*(k+1)-1
                    test_grid[j,k] = sum(test[j][start:fin]) / col_width
    
            # find the minimum value of test_grid. If minimum value is less than threshold, report j,k.
            if (np.min(test_grid) < THRESH): # i.e. on average, maximum of 1 pixel difference
                x,y = np.where(test_grid == np.min(test_grid))
                if layout == 'cabrera':
                    reference.append(format_6x2_cab[x[0]][y[0]]) # get corresponding label
                else:    
                    reference.append(format_6x2[x[0]][y[0]]) # get corresponding label
    
    return reference