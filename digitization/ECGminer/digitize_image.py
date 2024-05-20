"""
Entry point to ecg-miner digitization code
"""
import cv2
import numpy as np
from digitization.ECGminer.assets.Image import Image
from digitization.ECGminer.ECGClass import PaperECG

def digitize_image(restored_image, gridsize, sig_len=1000):
    # incoming: bad code~~~~~
    # convert greyscale to rgb
    restored_image = cv2.merge([restored_image,restored_image,restored_image])
    restored_image = np.uint8(restored_image)
    restored_image = Image(restored_image) # cleaned_image = reconstruction.Image.Image(cleaned_image)

    paper_ecg = PaperECG(restored_image, gridsize, sig_len=sig_len)
    ECG_signals, trace, raw_signals = paper_ecg.digitise()

    return ECG_signals, trace