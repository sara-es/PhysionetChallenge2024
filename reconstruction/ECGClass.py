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
                 gridsize) -> None:  # this image: Image thing is bad but fits w/ challenge function structure for now
        """
        Initialization of the ECG.

        Args:
            image (Image): Image of the ECG. (Image is the class from Image.py, not the filename of the image.)
        """

        self.image = image
        self.gridsize = gridsize

    def digitise(self) -> pd.DataFrame:
        """
        Digitize the ECG in paper format.

        Returns:
            DigitalECG: Digitized ECG.
        """

        ecg_crop, rect = self.preprocessor.preprocess(self.image)

        raw_signals = self.signal_extractor.extract_signals(ecg_crop)
        digitised_signals, trace = self.postprocessor.postprocess(self.gridsize, raw_signals, ecg_crop)
        trace.save("trace.png")

        return digitised_signals


# if __name__ == "__main__":
#     test_img = cv.imread("../tiny_testset/records100/00001_lr-0.png")
#     test_img = Image(test_img)
#     ecg = PaperECG(test_img)
#     digital_ecg = ecg.digitise()
