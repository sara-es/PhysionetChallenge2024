# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:42:14 2024

@author: hssdwo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:15:21 2024

@author: hssdwo
"""
import numpy as np
from scipy.signal import find_peaks
import pickle
import matplotlib.pyplot as plt
from digitization.ECGminer.assets.Point import Point

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
        #optional : plot output
        if plot_output:
            output = np.ones((test_im.shape[0], test_im.shape[1]))
            for i in range(len(signal_flat)):
                x = signal_flat[i][0]
                y = signal_flat[i][1]
                output[x,y] = 0
            plt.figure()
            plt.imshow(output) 
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
            


file = "C:/Users/hssdwo/Downloads/images.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)

print('extract signal and output in Point format')
test_im = data[0]
signals, rois = extract_rows(test_im, thresh = 50, plot_output = False)
raw_s = signal_to_miner(signals, rois)

print('turn raw signals into a matrix of signals')

#TODO: FIXME - some error in here
signal_arr = point_to_matrix(raw_s)
plt.plot(signal_arr[0,:])
plt.plot(signal_arr[1,:])
plt.plot(signal_arr[2,:])
plt.plot(signal_arr[3,:])
plt.show()


NROWS = len(signals)

#THRESH is a user parameter
THRESH = 1

#2.for each signal, get its np.diff - this converts from pixel coordinates to relative coordinates
diff_signals = np.diff(signal_arr)
# reference = []

from scipy.signal import correlate, correlation_lags
NCOLS = 4
ref = np.size(diff_signals,1)
col_width = ref//NCOLS

print('correlation between positions {pos1: [0,2], pos2: [0,1]} standard:{lead III, lead aVR}, cabrera:{lead -aVR, leadII')

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
print(a)
print(b)

if a>b:
    print('standard')
else:
    print('cabrera')