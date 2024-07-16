# A paper ECG is an image with... leads, signals, metadata, filename?

from digitization.ECGminer.Postprocessor import Postprocessor
from digitization.ECGminer.Preprocessor import Preprocessor
from digitization.ECGminer.SignalExtractor import SignalExtractor
from digitization.ECGminer.LayoutDetector import LayoutDetector 
from digitization.ECGminer.assets.Image import Image
import pandas as pd


class PaperECG:
    """
    Class to represent an ECG in paper format.
    """
    # path: str #useful to pass this here but for now we'll just use the image object
    image: Image
    preprocessor = Preprocessor()
    signal_extractor = SignalExtractor(n=4) #hardcoded n=4 for now because constant layout+rhythm
    layout_detector = LayoutDetector()
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
        signal_coords = self.signal_extractor.extract_signals(ecg_crop)
        print(signal_coords)
        
        ## DW: replace extract_signals with our own version.

        # check if reference pulses are present and where
        raw_signals, ref_pulse_present = self.layout_detector.detect_reference_pulses(signal_coords)

        # returns array of digitized signals and trace of cleaned image
        digitised_signals, trace = self.postprocessor.postprocess(
                self.gridsize, signal_coords, ecg_crop, self.sig_len, ref_pulse_present
            )

        return digitised_signals, trace, raw_signals
    
    def digitise_unet(self) -> pd.DataFrame:
        """
        Digitize the ECG in paper format.

        Returns:
            DigitalECG: Digitized ECG.
        """

        # returns image object
        ecg_crop, rect = self.preprocessor.preprocess(self.image)
        
        ## DW: replace extract_signals with our own version.
        # returns x and y coordinates of the traces in order
        # raises DigitizationError if failure occurs.
        signal_coords = self.signal_extractor.get_signals_dw(ecg_crop)
        
        # check if reference pulses are present and where
        raw_signals, ref_pulse_present = self.layout_detector.detect_reference_pulses(signal_coords)

        # returns array of digitized signals and trace of cleaned image
        digitised_signals, trace = self.postprocessor.postprocess(
                self.gridsize, signal_coords, ecg_crop, self.sig_len, ref_pulse_present
            )

        return digitised_signals, trace, raw_signals

