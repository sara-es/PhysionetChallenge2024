import numpy as np
import scipy as sp
from typing import Iterable
from digitization.ECGminer.assets.DigitizationError import DigitizationError, SignalExtractionError
from digitization.ECGminer.assets.Image import Image
from digitization.ECGminer.assets.Point import Point
from utils import constants


def get_roi(ecg: Image, n: int) -> Iterable[int]:
    """
    Get the coordinates of the ROI of the ECG image.

    Args:
        ecg (Image): ECG image from which to extract the ROI.
        n (int): Number of ROI to extract, ie the expected number of rows of ecg signal.

    Raises:
        DigitizationError: The indicated number of ROI could not be detected.

    Returns:
        Iterable[int]: List of row coordinates of the ROI.
    """
    WINDOW = 10
    SHIFT = (WINDOW - 1) // 2
    stds = np.zeros(ecg.height)
    for i in range(ecg.height - WINDOW + 1):
        x0, x1 = (0, ecg.width)
        y0, y1 = (i, i + WINDOW - 1)
        std = ecg[y0:y1, x0:x1].reshape(-1).std()
        stds[i + SHIFT] = std
    # Find peaks
    min_distance = int(ecg.height * 0.1)
    peaks, _ = sp.signal.find_peaks(stds, distance=min_distance)
    rois = sorted(peaks, key=lambda x: stds[x], reverse=True)
    if len(rois) < n:
        raise DigitizationError("The indicated number of rois could not be detected.")
    rois = rois[0: n]
    rois = sorted(rois)
    return rois


def find_start_coords(image, rois, row):
    """
    Find the start position of the ECG signal for each row.
    N.B. This will fail if signal from an adjacent row gets close to the baseline for this row.
    """
    # assume that the starting point is within 0.75 * gap between lines. This may not be true if 
    # signal starts on a particularly high/low QRS complex
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
    
    # find the start row - the black pixel that is nearest to the baseline. This also doesn't work
    # if adjacent row bleeds into this row
    rowvals = image[:,startcol]
    idxs = np.where(rowvals == 0)[0]
    idx = np.argmin(abs(idxs - row))
    startrow = idxs[idx]
    return startrow, startcol, endcol, row


def find_start_coords_from_bounds(image, bound_box, max_horizontal_bounds):
    """
    takes in an image and signal bounding box and returns the start row and column of signal
    bounding box is x_center, y_center, width, height in percent of image height, width
    """
    row = bound_box[1]*image.shape[0] # row y center
    top = int(row - image.shape[0]*bound_box[3]/2)
    bottom = int(row + image.shape[0]*bound_box[3]/2)
    if top < 0:
        top = 0
    if bottom > image.shape[0]:
        bottom = image.shape[0]
    row = int(row)

    left = int(image.shape[1]*bound_box[0] - image.shape[1]*bound_box[2]/2)
    right = int(image.shape[1]*bound_box[0] + image.shape[1]*bound_box[2]/2)
    if left < 0:
        left = 0
    if right > image.shape[1]:
        right = image.shape[1]

    # find the starting and end column - columns with black pixels within the active region
    colsums = np.min(image[top:bottom,:],0)
    idxs = np.where(colsums == 0)[0]
    startcol = idxs[0]
    endcol = idxs[-1]

    if startcol < max_horizontal_bounds[0]: # restrict starting point to edge of bounding box
        startcol = left
    if endcol > max_horizontal_bounds[1]: # restrict end point to edge of bounding box
        endcol = right
    
    # find the start row - the black pixel that is nearest to the baseline. This also doesn't work
    # if adjacent row bleeds into this row
    rowvals = image[:,startcol]
    idxs = np.where(rowvals == 0)[0]
    while idxs.size == 0: # if there is no black pixel in this column, move to the next column
        startcol += 1
        rowvals = image[:,startcol]
        idxs = np.where(rowvals == 0)[0]
    idx = np.argmin(abs(idxs - int(row)))
    startrow = idxs[idx]

    return startrow, startcol, endcol, row


def signal_to_miner(signals, rois):
    """
    Convert ecg sparse matrices into ecg miner format.
    """
    all_sigs = []
    for i, signal in enumerate(signals):
        if signal == []: # erroneous bounding box, probably
            continue
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


def ecg_sqi(signals, rois, max_duration=10, thresh_lens=5, thresh_secs=0.1) -> bool:
    ''' reconstruction signal quality based on two criteria:
    1. Are the rows of ECG all roughly the same length
    2. Do the rows of ECG mainly stick to a single baseline?

    Args:
        signals: list of lists of Point objects, each sub-list represents a row of ECG
        rois: list of ints, the y-coordinates of the rows of ECG
        thresh_lens: int, set consistency (in pixels) between rows. i.e. reject if row length 
            are different
        thresh_secs: float, set the maximum amount of time we allow a signal to be off baseline

    returns:
        SQI: bool, True if signal quality is good, False if signal quality is bad
    '''
    SQI = True
    thresh = len(signals[0])/(max_duration/thresh_secs)

    sig_len = []
    for signal in signals:
        sig_len.append(len(signal))
        
    if (np.max(sig_len) - np.min(sig_len)) > thresh_lens:
        SQI = False

    for i, signal in enumerate(signals):
        signal_y = [p.y for p in signal]
        filt_signal = []
        for j in range(len(signal)):
            #replace y value with nearest roi
            idx = np.where(np.abs(rois - signal_y[j]) == np.min(np.abs(rois - signal_y[j])))
            idx = idx[0][0]
            filt_signal.append(bool(rois[idx] - rois[i]))

        # count the longest contiguous section of non-zeros
        section = 0
        this_section = 0
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
    return SQI


def extract_row_signals(ecg: Image, yolo_rois_cropped : np.array, n_lines: int) -> Iterable[Iterable[Point]]:
    x_coords, y_coords, endcols, rows = [], [], [], []
    if yolo_rois_cropped.size > 0 and constants.YOLO_ROIS and constants.YOLO_CONFIG == 'yolov7-ecg-1c':
        bounding_boxes = yolo_rois_cropped[yolo_rois_cropped[:, 1].argsort()] # sort by y_center
        rois = sorted((ecg.height*bounding_boxes[:, 1]).astype(int)) # y_center of bounding box
        # get max horizontal (left and right) bounds
        horizontal_bounds = np.zeros(2)
        min_left = np.min(ecg.width*bounding_boxes[:, 0] - ecg.width*bounding_boxes[:, 2]/2)
        max_right = np.max(ecg.width*bounding_boxes[:, 0] + ecg.width*bounding_boxes[:, 2]/2) 
        horizontal_bounds[0] = min_left if min_left > 0 else 0
        horizontal_bounds[1] = max_right if max_right < ecg.width else ecg.width
        for row in bounding_boxes:
            x, y, endcol, row_yval = find_start_coords_from_bounds(ecg.data, row, horizontal_bounds)
            x_coords.append(x)
            y_coords.append(y)
            endcols.append(endcol)
            rows.append(row_yval)
    else:
        rois = get_roi(ecg, n_lines)
        if constants.YOLO_ROIS:
            print(f"No yolo boxes found, default ROIs: {rois}")
        for row in rois:
            x, y, endcol, row_yval = find_start_coords(ecg.data, rois, row)
            x_coords.append(x)
            y_coords.append(y)
            endcols.append(endcol)
            rows.append(row_yval)

    thresh = 50
    test_im = ecg.data
    s = []
    
    for x, y, endcol, row in zip(x_coords, y_coords, endcols, rows):
        signal = []
        signal_col = []
        signal_col.append([x,y])
        
        while y < endcol:
            # search up
            x_idx = x-1
            if x_idx < 0:
                raise SignalExtractionError("Signal extraction attemped to leave image bounds.")
            while test_im[x_idx, y] == 0:
                signal_col.append([x_idx,y])
                x_idx = x_idx-1
                if x_idx == 0:
                    break
            # search down
            x_idx = x+1
            if x_idx >= test_im.shape[0]:
                raise SignalExtractionError("Signal extraction attemped to leave image bounds.")
            while test_im[x_idx, y] == 0:
                signal_col.append([x_idx,y])
                x_idx = x_idx+1
                if x_idx == test_im.shape[0] - 1:
                    break
        
            signal_col = sorted(signal_col, key=lambda x: x[0], reverse=False)
            signal.append(signal_col)
            
            # go to next column and find nearest black pixels
            y = y+1
            match = 0
            dists_fails = 0 # count the number of times we fail to find a match
            while match == 0:                
                # get distances between possible matches
                dists = []
                candidates_rep = []
                signal_col_temp = [i[0] for i in signal_col]
                candidates = np.where(test_im[:,y] == 0)[0]
                for i in signal_col_temp:
                    temp_dist = np.subtract(candidates, i)
                    dists.append(temp_dist)
                    candidates_rep.append(candidates)
                dists= [x for xs in dists for x in xs]
                dists = np.array(dists)
                
                candidates = [x for xs in candidates_rep for x in xs]
                candidates = np.array(candidates)

                if len(dists) == 0: # no candidates for matches at all
                    dists_fails += 1 
                    if dists_fails > thresh:
                        raise SignalExtractionError("Could not match signal pixels: bad image or ROI.")
                
                if len(dists) > 0 and np.min(abs(dists))<thresh:                    
                    #bit of voodoo here to try to make sure that we favour returning towards 
                    # baseline when there is a choice
                    dist_idx = np.where(abs(dists) == min(abs(dists)))[0]

                    if len(dist_idx)>1:
                        # if there is more than one minimum, select the one that takes us closer 
                        # to the baseline
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
                else:
                    #check whether ecg has returned to near baseline (new lead), if so, reset x_idx
                    check_col = test_im[(row-thresh):(row+thresh),y+1]
                    x = np.where(check_col == 0)[0]
                    if len(x) > 0:
                        x_idx = row-thresh+x[0]
                        x = x_idx
                        match = 1
                    else: #skip a column and try again
                        y = y+1
  
        #sanitise the list
        signal_flat = [x for xs in signal for x in xs]
        s.append(signal_flat)

    point_signal = signal_to_miner(s, rois)
    return point_signal, rois