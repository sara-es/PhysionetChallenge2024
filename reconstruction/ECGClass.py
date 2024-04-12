# A paper ECG is an image with... leads, signals, metadata, filename?
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from reconstruction.Lead import Lead
from reconstruction.Postprocessor import Postprocessor
from reconstruction.Preprocessor import Preprocessor
from reconstruction.SignalExtractor import SignalExtractor
from reconstruction.Image import Image
import matplotlib.pyplot as plt
import pandas as pd


class PaperECG:
    """
    Class to represent an ECG in paper format.
    """
    # path: str #useful to pass this here but for now we'll just use the image object
    image: Image
    preprocessor = Preprocessor()
    signal_extractor = SignalExtractor(n=4) #hardcoded n=4 for now because constant layout+rhythm
    postprocessor = Postprocessor()

    def __init__(self,
                 image: Image,
                 gridsize,
                 sig_len : int) -> None:  # this image: Image thing is bad but fits w/ challenge function structure for now
        """
        Initialization of the ECG.

        Args:
            image (Image): Image of the ECG. (Image is the class from Image.py, not the filename of the image.)
        """

        self.image = image
        self.gridsize = gridsize
        self.sig_len = sig_len

    def digitise(self) -> pd.DataFrame:
        """
        Digitize the ECG in paper format.

        Returns:
            DigitalECG: Digitized ECG.
        """

        # returns image object
        ecg_crop, rect = self.preprocessor.preprocess(self.image)

        # returns x and y coordinates of the traces in order
        raw_signals = self.signal_extractor.extract_signals(ecg_crop)

        # returns array of digitized signals and trace of cleaned image
        digitised_signals, trace = self.postprocessor.postprocess(self.gridsize, raw_signals, ecg_crop, self.sig_len)

        return digitised_signals, trace
