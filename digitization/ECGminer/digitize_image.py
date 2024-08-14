"""
Entry point to ecg-miner digitization code
"""
import cv2
import numpy as np
import pandas as pd
from digitization.ECGminer.assets.Image import Image
from digitization.ECGminer.Preprocessor import Preprocessor
from digitization.ECGminer import extract_signals, vectorize_signals


def get_ECG_rows(yolo_boxes):
    yolo_boxes = np.array(yolo_boxes)
    #get the number of long leads
    long_leads = sum(yolo_boxes[:,0])
    num_rows = 3 + long_leads #simplifying challenge assumption that the short leads are in 3x4 format.
    # n.b. to generalise this, we need to work out the number of short lead rows - we could do this by merging rows in the x direction.
    
    # get total bounding box
    left = min(yolo_boxes[:, 1] - yolo_boxes[:, 3]/2)
    right = max(yolo_boxes[:, 1] + yolo_boxes[:, 3]/2)
    top = min(yolo_boxes[:, 2] - yolo_boxes[:, 4]/2)
    bottom = max(yolo_boxes[:, 2] - yolo_boxes[:, 4]/2)
   
    # note bounding box is in percentage of image size
    bounding_box = [left, right, top, bottom]
    
    return num_rows, bounding_box


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
    # Invert the colours
    restored_image = abs(restored_image - 1)*255
    # convert greyscale to rgb
    restored_image = cv2.merge([restored_image,restored_image,restored_image])
    restored_image = np.uint8(restored_image)
    restored_image = Image(restored_image)

    # Convert YOLO xywh boxes to ROIs
    num_rows, bounding_box = get_ECG_rows(yolo_rois)

    # returns image object
    preprocessor = Preprocessor()
    ecg_crop, rect = preprocessor.preprocess(restored_image)
    
    ## DW: replace extract_signals with our own version.
    # returns x and y coordinates of the traces in order
    # raises DigitizationError if failure occurs.
    signal_coords, rois = extract_signals.extract_row_signals(ecg_crop, n_lines=int(num_rows))

    # check for reference pulses, then convert to digitized signals
    # returns array of digitized signals, original signal coordinates, and gridsize
    # gridsize: float, scaling factor for the signals in pixel units
    digitised_signals, raw_signals, gridsize = vectorize_signals.vectorize(signal_coords, 
                                                                sig_len, max_duration,
                                                                is_generated_image)
    
    return digitised_signals, raw_signals, gridsize

