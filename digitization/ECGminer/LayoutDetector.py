# Standard library imports
from typing import Iterable, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from scipy import interpolate

# Application-specific imports
from digitization.ECGminer.assets.Point import Point
from digitization.ECGminer.assets.Lead import Lead
from digitization.ECGminer.assets.DigitizationError import DigitizationError
from digitization.ECGminer.assets.Format import Format


class LayoutDetector:
    """
    Detects reference pulses and their positions in the ECG.

    Can possibly extend to detect ECG layout (number and position of leads, etc.) if needed.
    """

    def __init__(
        self
    ) -> None:
        """
        Initialization of the layout detector.
        """
        pass

    def detect_reference_pulses(
        self, raw_signals: Iterable[Iterable[Point]],
    ) -> Tuple[np.array, bool]:
        signals, ref_pulse_present = self.__vectorize(raw_signals)
        return signals, ref_pulse_present

    def __vectorize(
        self,
        raw_signals: Iterable[Iterable[Point]],
    ) -> Tuple[np.array, int]:
        """
        Vectorize the signals, returning only raw coordinate (pixel) values, then check if reference 
        pulses are present in the ECG signals. Based on pulse_pos from Dave's work in pulse_detect.py.

        Assumes signals are in format (2, 4, N) where the first dimension corresponds to 
        (x, y) coordinates, 4 is the number of rows, and N is the length of the signal (number of samples)

        Returns:
            np.array: Vectorized signals.
            ref_pulse_present: int, 0 if no reference pulses, 1 if present on left, 2 if present at right
        """

        signal_slice = slice(0, None)
        signals = [rs[signal_slice] for rs in raw_signals]
        max_len = max(map(lambda signal: len(signal), signals))
        raw_signals = np.zeros((len(signals), max_len))
        pulses =  np.zeros(len(signals))

        for i in range(len(signals)):
            signal = [-p.y for p in signals[i]] # invert y coords (coords *down* from top)
            # interpolate to linear x coords
            interpolator = interpolate.interp1d(np.arange(len(signal)), signal) # current x, y
            raw_signals[i, :] = interpolator(
                np.linspace(0, max_len-1, max_len) # xnew, returns ynew
            ) 
            pulses[i] = (self.__row_pulse_pos(raw_signals[i, :]))

        if sum(pulses == 1) >= 2:
            return raw_signals, 1
        elif sum(pulses == 2) >= 2:
            return raw_signals, 2
        else:
            return raw_signals, 0
    

    def __is_pulse(
        self,
        signal: np.array,
        baseline: float,
        min_width: int = 20,
        min_height: int = 25,
        tol: int = 3,
    ) -> bool:
        """
        Copy of is_pulse from Dave's work in pulse_detect.py
        """
        for i in range(len(signal)-min_width):
            segment = signal[i:i+(min_width)]
            seg_range = max(segment)-min(segment)
            seg_median = np.median(segment)
            offset = seg_median - baseline
            if (seg_range < tol) & (offset > min_height):
                return True
        return False


    def __row_pulse_pos(
        self,
        signal: np.array,
        scan_perc : float = 0.1,
    ) -> int:
        """

        """
        baseline = np.median(signal)
        scan_length = round(len(signal) * scan_perc) # look in the first and last 10% of the signal
        # get left hand segment
        left_segment = signal[0:scan_length]
        right_segment = signal[-scan_length:]
        is_left = self.__is_pulse(left_segment, baseline)
        is_right = self.__is_pulse(right_segment, baseline)

        if is_left & is_right:
            return 0 # assume there's been a mistake, no reference pulses 
        elif is_left:
            return 1 
        elif is_right:
            return 2
        else:
            return 0 # will be False if bool
        