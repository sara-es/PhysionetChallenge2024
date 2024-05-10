# Standard library imports
from typing import Iterable, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from scipy import interpolate

# Application-specific imports
from reconstruction.Point import Point
from reconstruction.Lead import Lead
from reconstruction.DigitizationError import DigitizationError
from reconstruction.Format import Format


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
        self, raw_signals: Iterable[Iterable[Point]]
    ) -> Tuple[np.array, bool]:
        signals = self.__vectorize(raw_signals)
        ref_pulse_present = self.__detect_reference_pulses(signals)
        return signals, ref_pulse_present

    def __vectorize(
        self,
        raw_signals: Iterable[Iterable[Point]],
    ) -> np.array:
        """
        Vectorize the signals, returning only raw coordinate (pixel) values.
        """

        signal_slice = slice(0, None)
        signals = [rs[signal_slice] for rs in raw_signals]
        # assume length (independent axis) is difference between first and last x coordinate
        max_len = max(map(lambda signal: len(signal), signals))
        raw_signals = np.zeros((len(signals), max_len))

        for i in range(len(signals)):
            signal = [-p.y for p in signals[i]] # invert y coords (coords *down* from top)
            x_coords = [p.x for p in signals[i]]
            max_obs = x_coords[-1] - x_coords[0] 
            # interpolate to linear x coords
            interpolator = interpolate.interp1d(np.arange(len(signal)), signal) # current x, y
            raw_signals[i, :] = interpolator(
                np.linspace(0, max_len-1, max_len) # xnew, returns ynew
            )  
        return raw_signals
    
    def __find_pulse(
        self,
        xdiff: np.array,
        start_idx: int,
        tol: int = 3,
        min_height: int = 25,
        min_width: int = 25,
        max_width: int = 100,
    ) -> bool:
        """
        find_pulse from Dave's testing
        """
        idx = start_idx
        up = -1
        down = -1
        up_y = -1
        
        while idx < (start_idx + max_width):
            if up == -1:
                if xdiff[idx] > min_height:
                    up = idx # we expect this to be at the very start of the signal (i.e. idx == 0), but might not be if the digitisation is slightly wrong
                    up_y = xdiff[idx]
            else:  # if an up pulse has been found, then we are either on a false up, a plateau, or about to hit a down pulse 
                if (idx-up) > min_width: #min width of pulse we will accept
                    if abs(xdiff[idx] + up_y) < tol: # can make this constraint tighter to ensure that the down matches the up
                        down = idx #success!
                else:
                    if abs(xdiff[idx]) > tol: # if we actually aren't on a plateau, then this is not a square wave
                        up = -1 # reset the squarewave
                        up_y = -1
            idx = idx + 1

        return up, down

    def __detect_reference_pulses(
        self,
        raw_signals: np.array,
        max_width: int = 100,
    ) -> Tuple[int, int]:
        """
        Detect reference pulses in the ECG signals.
        Can use this to allow for Postprocessor() to look for reference pulses.
        Assumes signals are in format (2, 4, N) where the first dimension corresponds to 
        (x, y) coordinates, 4 is the number of rows, and N is the length of the signal (number of samples)
        """
        ups = []
        downs = []

        for signal in raw_signals:
            xdiff = np.diff(signal)

            # import matplotlib.pyplot as plt
            # plt.plot(xdiff)
            # plt.show()
            # plt.close()

            # check for square wave on left, middle and right of signal
            max_width = 100
            l_up, l_down = self.__find_pulse(xdiff,0, max_width=max_width) #check left
            m_up, m_down = self.__find_pulse(xdiff,len(xdiff)//2-max_width//2, max_width=max_width) #check middle
            r_up, r_down = self.__find_pulse(xdiff,len(xdiff)-max_width, max_width=max_width)

            # todo - add sanity checking here - square wave should only exist in one place

            # return the non -1 indices
            up = max([l_up, m_up, r_up])
            down = max([l_down, m_down, r_down])
            baseline = signal[up] # TODO - check that this is the right index
            #RETURN: the pixel value of the bottom of the pulse (to get 0mV), start of pulse, end of pulse
            ups.append(up)
            downs.append(down)

        ## ---- each row will return an up and down index. Check for consistency ----
        if np.array(ups).mean() == -1 and np.array(downs).mean() == -1:
            # no reference pulse found
            return False
        else:
            tol = 2
            nonzero_ups = np.array(ups)[np.where(np.array(ups) > 0)]
            nonzero_downs = np.array(downs)[np.where(np.array(downs) > 0)]

            print(nonzero_ups, nonzero_downs)
            # consensus agreement - if at least two sets of indices are within tol, then accept the pulse
            if len(nonzero_ups) > 1 and len(nonzero_downs) > 1:
                if (np.max(nonzero_ups) - np.min(nonzero_ups)) < tol & (np.max(nonzero_downs) - np.min(nonzero_downs)) < tol:
                    return True
            # pulse index might be slightly different (if rotation isn't perfect), so take average if needed

        return False