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



@dataclass
class DigitalECG:
    """
    Class to represent an ECG.
    """
    signals: dict[Lead, np.ndarray]
    metadata: dict[str, str]
    filename: str

    def save(self, path: str) -> None:
        """
        Save as .dat file.
        """
        pass

    def plot(self) -> None:
        """
        Plot the ECG.
        """
        pass

    def get_lead(self, lead: Lead) -> np.ndarray:
        """
        Get the signal of a lead.

        Args:
            lead (Lead): Lead.

        Returns:
            np.ndarray: Signal.
        """
        return self.signals[lead]


class PaperECG:
    """
    Class to represent an ECG in paper format.
    """
    path: str
    image: Image
    preprocessor = Preprocessor()
    signal_extractor = SignalExtractor(n=1)
    postprocessor = Postprocessor()

    def __init__(self, filename: str) -> None:
        """
        Initialization of the ECG.

        Args:
            filename (str): Filename of the ECG.
        """

        try:
            self.image = Image(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f'File "{filename}" does not exist')

    def digitise(self) -> DigitalECG:
        """
        Digitize the ECG in paper format.

        Returns:
            DigitalECG: Digitized ECG.
        """
        frame = self.image.copy
        ecg_crop, rect = self.preprocessor.preprocess(self.image)

        raw_signals = self.signal_extractor.extract_signals(ecg_crop)
        data, trace = self.postprocessor.postprocess(raw_signals, ecg_crop)


        return DigitalECG({}, {}, "")

    def save(self, path: str) -> None:
        """
        Save as .png file.
        """
        pass

    def plot(self) -> None:
        """
        Plot the ECG.
        """
        pass

    def __detect_grid(self) -> np.ndarray:
        """
        Detect the grid of the ECG.

        Returns:
            np.ndarray: Image of the grid.
        """
        pass

    def __rotate(self, angle: float) -> None:
        """
        Rotate the ECG.

        Args:
            angle (float): Angle of rotation.
        """
        pass

    def __crop(self, rect: tuple[tuple[int, int], tuple[int, int]]) -> None:
        """
        Crop the ECG.

        Args:
            rect (tuple[tuple[int, int], tuple[int, int]]): Rectangle to crop.
        """
        pass

    def __binarize(self) -> None:
        """
        Binarize the ECG.
        """
        pass

    def __remove_gridlines(self) -> None:
        """
        Remove the gridlines of the ECG.
        """
        pass

    def __extract_signals(self) -> dict[Lead, np.ndarray]:
        """
        Extract the signals of the ECG.

        Returns:
            dict[Lead, np.ndarray]: Signals.
        """
        pass



if __name__ == "__main__":
    ecg = PaperECG("../tiny_testset/records100/00001_lr-0.png")
    digital_ecg = ecg.digitise()
