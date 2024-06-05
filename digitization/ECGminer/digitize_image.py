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

" digitize_image_unet takes the output of the u-net. ECG_signals are extracted with an alternative method: PaperECG/digitize_unet"
def digitize_image_unet(restored_image, gridsize, sig_len=1000):
    # Invert the colours
    restored_image = abs(restored_image - 1)*255
    # convert greyscale to rgb
    restored_image = cv2.merge([restored_image,restored_image,restored_image])
    restored_image = np.uint8(restored_image)
    restored_image = Image(restored_image) # cleaned_image = reconstruction.Image.Image(cleaned_image)

    paper_ecg = PaperECG(restored_image, gridsize, sig_len=sig_len)
    ECG_signals, trace, raw_signals = paper_ecg.digitise_unet()

    return ECG_signals, trace