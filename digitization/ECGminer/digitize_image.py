"""
Entry point to ecg-miner digitization code
"""
import cv2
import numpy as np
from digitization.ECGminer.assets.Image import Image
from digitization.ECGminer.ECGClass import PaperECG


def digitize_image(restored_image, gridsize, sig_len=1000):
    # convert greyscale to rgb
    restored_image = cv2.merge([restored_image,restored_image,restored_image])
    restored_image = np.uint8(restored_image)
    restored_image = Image(restored_image) # cleaned_image = reconstruction.Image.Image(cleaned_image)

    paper_ecg = PaperECG(restored_image, gridsize, sig_len=sig_len)
    ECG_signals, trace, raw_signals = paper_ecg.digitise()

    return ECG_signals, trace


def digitize_image_unet(restored_image, gridsize, sig_len=1000):
    """ 
    digitize_image_unet takes the output of the u-net. ECG_signals are extracted with an 
    alternative method: PaperECG/digitize_unet

    Input - an image (mask) from the unet, where ECG = 1, background - 0
    Output - reconstructed ECG signals, trace of 
       
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
    restored_image = Image(restored_image) # cleaned_image = reconstruction.Image.Image(cleaned_image)

    paper_ecg = PaperECG(restored_image, gridsize, sig_len=sig_len)
    ECG_signals, trace, raw_signals = paper_ecg.digitise_unet()

    return ECG_signals, trace