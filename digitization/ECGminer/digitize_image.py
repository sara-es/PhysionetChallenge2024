"""
Entry point to ecg-miner digitization code
"""
import cv2
import numpy as np
#import pandas as pd
from digitization.ECGminer.assets.Image import Image
from digitization.ECGminer import preprocess, extract_signals, vectorize_signals


def digitize_image_unet(restored_image, yolo_rois, sig_len=1000, max_duration=10, is_generated_image=True):
    """ 
    digitize_image_unet takes the output of the u-net

    Input - an image (mask) from the unet, where ECG = 1, background - 0
    Output - reconstructed ECG signals, trace of, quality index, grid size in pixels
       
    There are three stages:
        1. convert unet image into a set of Point (ecg-miner) objects that correspond to each row of ECG, in pixels
        2. identify and remove any reference pulses
        3. convert into individual ECG channels and convert from pixels to mV
    
    """
    # Convert YOLO xywh boxes to ROIs
    num_rows, box = preprocess.get_ECG_rows(yolo_rois)
    if is_generated_image and num_rows > 4:
        num_rows = 4
    
    restored_image, bounding_rect = preprocess.crop_image(restored_image, box)
   
    ## DW: replace extract_signals with our own version.
    # returns x and y coordinates of the traces in order
    # raises DigitizationError if failure occurs.
    signal_coords, rois = extract_signals.extract_row_signals(restored_image, n_lines=int(num_rows))

    # check for reference pulses, then convert to digitized signals
    # returns array of digitized signals, original signal coordinates, and gridsize
    # gridsize: float, scaling factor for the signals in pixel units
    digitised_signals, raw_signals, gridsize = vectorize_signals.vectorize(signal_coords, 
                                                                sig_len, max_duration,
                                                                is_generated_image)
    
    trace = {}
    trace['raw_signals']=raw_signals
    trace['bounding_rect']=bounding_rect
    
    return digitised_signals, trace, gridsize

