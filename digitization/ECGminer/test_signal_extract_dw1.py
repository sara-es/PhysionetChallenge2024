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

file = "C:/Users/hssdwo/Downloads/images.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)

test_im = data[5]

test_im = abs(test_im - 1)*255
plt.imshow(test_im)
height = test_im.shape[0]
width = test_im.shape[1]
rois = get_roi(test_im)

# assume that the starting point is somewhere close to the baseline
roi_gap = int(np.max(np.diff(rois)) *0.75)
# assume that starting point for a row is the closest black pixel to the roi centre
#rois = [rois[1]]

for row in rois:
    #row = rois[0]
    top = row - roi_gap
    if top < 0:
        top = 0
    bottom = row + roi_gap
    if bottom > height:
        bottom = height
    
    # find the starting and end column - areas with black pixels
    colsums = np.min(test_im[top:bottom,:],0)
    idxs = np.where(colsums == 0)[0]
    startcol = idxs[0]
    endcol = idxs[-1]
    
    # find the start row - the black pixel that is nearest to the roi
    rowvals = test_im[:,startcol]
    idxs = np.where(rowvals == 0)[0]
    idx = np.argmin(abs(idxs - row))
    startrow = idxs[idx]
    
    # we now have a starting point [startrow, startcol]
    # the logic is as follows:
    #   - search up and down to find the ends of the lines
    #   - from the line ends, find nearest point in the next column (assume there is only one, as unet is good)
    #   - repeat until the end
    
    x = startrow
    y = startcol
    
    signal = []
    signal_col = []
    signal_col.append([x,y])
    thresh = 50 #maximum allowable jump between columns
    
    while y < endcol:
        # line search
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
                #dist_idx = np.argmin(abs(dists))
                #dist_idx = np.where(abs(dists))
                #if y == 1614:
                #    print(np.where(abs(dists) == min(abs(dists))))
                
                #bit of voodoo here to try to make sure that we favour returning towards baseline when there is a choice
                dist_idx = np.where(abs(dists) == min(abs(dists)))[0]
                if len(dist_idx)>1:
                    #if there is more than one minimum, select the one that takes us closer to the baseline
                    x = candidates[dist_idx]
                    i = np.argmin(abs(x-row))
                    x = x[i]
                else:
                    x = candidates[dist_idx]
                signal_col = []
                signal_col.append([x,y])
                match = 1
            elif y == endcol:
                match = 1
            else: #skip a column and try again
                y = y+1
                
    #sanitise the list
    signal_flat = [x for xs in signal for x in xs]
    
    #test - plot
    output = np.ones((height, width))
    for i in range(len(signal_flat)):
        x = signal_flat[i][0]
        y = signal_flat[i][1]
        output[x,y] = 0
    plt.figure()
    plt.imshow(output)
    
    