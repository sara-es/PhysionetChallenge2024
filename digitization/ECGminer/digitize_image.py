"""
Entry point to ecg-miner digitization code
"""
import cv2
import numpy as np
from digitization.ECGminer.assets.Image import Image
from digitization.ECGminer.Preprocessor import Preprocessor

from digitization.ECGminer import extract_signals, vectorize_signals


def digitize_image_unet(restored_image, sig_len=1000, max_duration=10, is_generated_image=True):
    """ 
    digitize_image_unet takes the output of the u-net

    Input - an image (mask) from the unet, where ECG = 1, background - 0
    Output - reconstructed ECG signals, trace of, quality index, grid size in pixels
       
    There are three stages:
        1. convert unet image into a set of Point (ecg-miner) objects that correspond to each row of ECG, in pixels
        2. identify and remove any reference pulses
        3. convert into individual ECG channels and convert from pixels to mV
    
    """
    #TODO: remove this fudge once the u-net is fixed
    restored_image[:,0] = 0
    restored_image[:,-1] = 0
    restored_image[0,:] = 0
    restored_image[-1,:] = 0
    
    # Invert the colours
    restored_image = abs(restored_image - 1)*255
    # convert greyscale to rgb
    restored_image = cv2.merge([restored_image,restored_image,restored_image])
    restored_image = np.uint8(restored_image)
    restored_image = Image(restored_image)

    # returns image object
    preprocessor = Preprocessor()
    ecg_crop, rect = preprocessor.preprocess(restored_image)
    
    ## DW: replace extract_signals with our own version.
    # returns x and y coordinates of the traces in order
    # raises DigitizationError if failure occurs.
    # hardcoded n_lines=4 for now because constant layout+rhythm
    # TODO: get rows of data from yolo
    signal_coords, rois = extract_signals.extract_row_signals(ecg_crop, n_lines=4)

    # check for reference pulses, then convert to digitized signals
    # returns array of digitized signals, original signal coordinates, and gridsize
    # gridsize: float, scaling factor for the signals in pixel units
    digitised_signals, raw_signals, gridsize = vectorize_signals.vectorize(signal_coords, 
                                                                sig_len, max_duration,
                                                                is_generated_image)

    return digitised_signals, raw_signals, gridsize

