import numpy as np
import scipy as sp
import pandas as pd
from typing import Iterable, Tuple
from digitization.ECGminer.assets.DigitizationError import DigitizationError
from digitization.ECGminer.assets.Point import Point
from digitization.ECGminer.assets.Format import Format
from digitization.ECGminer.assets.Lead import Lead


def is_pulse(
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


def row_pulse_pos(
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
    is_left = is_pulse(left_segment, baseline)
    is_right = is_pulse(right_segment, baseline)

    if is_left & is_right:
        return 0 # assume there's been a mistake, no reference pulses 
    elif is_left:
        return 1 
    elif is_right:
        return 2
    else:
        return 0 # will be False if bool


def remove_pulses(raw_signals: Iterable[Iterable[Point]], ref_pulse_present: int
                  ) -> Tuple[Iterable[Iterable[Point]], Iterable[Tuple[int, int]]]:
    """
    TODO Vintage ECGminer code - needs work
    """
    rp_at_right = ref_pulse_present == 2
    INI, MID, END = (0, 1, 2)
    LIMIT = min([len(signal) for signal in raw_signals])
    PIXEL_EPS = 38.5 # MAGIC NUMBER: 5mm in pixels + 1 in generated images, hardcoded for now
    # Check if ref pulse is at right side or left side of the ECG
    furthest_right_pixels = [pulso[-1].y for pulso in raw_signals]

    # TODO: no accounting for reference pulses at right currently
    direction = (
        range(-1, -LIMIT, -1) if rp_at_right else range(LIMIT)
    )
    pulse_pos = INI
    ini_count = 0
    cut = None
    for i in direction:
        y_coords = [
            pulso[i].y - ini
            for pulso, ini in zip(raw_signals, furthest_right_pixels)
        ]
        y_coords = sorted(y_coords)
        at_v0 = any([abs(y) <= PIXEL_EPS for y in y_coords])
        break_symmetry = (pulse_pos == END) and (
            not at_v0 or ini_count <= 0
        )
        if break_symmetry:
            cut = i
            break
        if pulse_pos == INI:
            if at_v0:
                ini_count += 1
            else:
                pulse_pos = MID
        elif pulse_pos == MID and at_v0:
            pulse_pos = END
            ini_count -= 1
        elif pulse_pos == END:
            ini_count -= 1
    try:
        # Slice signal
        signal_slice = (
            slice(0, cut + 1) if rp_at_right else slice(cut, None)
        )
        signals = [rs[signal_slice] for rs in raw_signals]
        # Slice pulses
        pulse_slice = (
            slice(cut + 1, None) if rp_at_right else slice(0, cut + 1)
        )
        ref_pulses = [
            sorted(map(lambda p: p.y, rs[pulse_slice]), reverse=True)
            for rs in raw_signals
        ]
        ref_pulses = [
            (furthest_right_pixels[i], ref_pulses[i][-1])
            for i in range(len(raw_signals))
        ]
    except TypeError:
        # unable to detect reference pulses, though they should be present
        signal_slice = (
            slice(None, None)
        )
        signals = [rs[signal_slice] for rs in raw_signals]

        ref_pulses = []
        
    return (signals, ref_pulses)


def vectorize(signal_coords: Iterable[Iterable[Point]], sig_len: int, max_duration: int
    ) -> Tuple[np.array, float, int]:
    """
    Vectorize the signals, normalizing them and storing them in a dataframe.
    Assumes signals are in format (2, 4, N) where the first dimension corresponds to 
    (x, y) coordinates, 4 is the number of rows, and N is the length of the signal (number of
    samples)
    
    1. Extract only raw coordinate (pixel) values 
    2. Check if reference pulses are present in the ECG signals (based on pulse_pos from Dave's
        work in pulse_detect.py)
    3. Remove them from pixel list if so with remove_pulses
    3. Calculate signal scaling factor (gridsize) from the median total number of pixels the 
    signal covers horizontally

    Args:
        signal_coords: Iterable[Iterable[Point]], x and y coordinates of the signals.
        sig_len : int, number of samples in the signal.
        max_duration: int, maximum duration of the signal in seconds

    Returns:
        np.array: Vectorized signals.
        gridsize: float, scaling factor for the signals.
        ref_pulse_present: int, 0 if no reference pulses, 1 if present on left, 2 if present 
            at right
    """

    signal_slice = slice(0, None)
    signals = [rs[signal_slice] for rs in signal_coords]
    max_len = max(map(lambda signal: len(signal), signals))
    raw_signals = np.zeros((len(signals), max_len))
    pulses =  np.zeros(len(signals))

    for i in range(len(signals)):
        signal = [-p.y for p in signals[i]] # invert y coords (coords *down* from top)
        # interpolate to linear x coords
        interpolator = sp.interpolate.interp1d(np.arange(len(signal)), signal, fill_value='extrapolate') # current x, y
        raw_signals[i, :] = interpolator(
            np.linspace(0, max_len-1, max_len) # xnew, returns ynew
        ) 
        pulses[i] = (row_pulse_pos(raw_signals[i, :]))

    # if we think we've found reference pulses in 2 or more rows, they're probably present
    if sum(pulses == 1) >= 2:
        ref_pulse_present = 1
    elif sum(pulses == 2) >= 2:
        ref_pulse_present = 2
    else:
        ref_pulse_present = 0

    if ref_pulse_present:
        vectorized_signals, ref_pulses = remove_pulses(signal_coords, ref_pulse_present)
    else: # Slice signal
        signal_slice = (
            slice(None, None)
        )
        vectorized_signals = [rs[signal_slice] for rs in signal_coords]

    # Now that we've stripped out the reference pulses, calculate grid size and re-interpolate to correct
    # number of samples (this could definitely be done more efficiently)

    #### START OF VECTORIZE
    # Pad all signals to closest multiple number of ECG ncols
    NROWS, NCOLS = (3, 4) # ATTENTION: LAYOUT HARD CODED FOR NOW
    ORDER = Format.STANDARD # ATTENTION: LAYOUT HARD CODED FOR NOW, alternative: Format.CABRERA
    rhythm_leads = [Lead.II] # ATTENTION: RHYTHM LEAD HARD CODED FOR NOW
    max_len = max(map(lambda signal: len(signal), vectorized_signals))
    max_diff = max_len % NCOLS
    max_pad = 0 if max_diff == 0 else NCOLS - max_diff

    grid_size_px = max_len / (max_duration/0.2) # 0.2 s per square

    # hack to force correct sampling frequency
    total_obs = int(sig_len)
    # Linear interpolation to get a certain number of observations
    interp_signals = np.empty((len(vectorized_signals), total_obs))
    for i in range(len(vectorized_signals)):
        signal = [p.y for p in vectorized_signals[i]]

        # ATTENTION
        # HACK BELOW
        # This adds a few pixels to the start of the signal, because the image-kit generator
        # overlaps the ecg and the reference pulses slightly. On real images, we should NOT do this!
        # if ref_pulse_generated == 1: # ref pulses at left, pad start of signal
        #     buffer = [signal[0]]*9
        #     signal = buffer + signal
        # if ref_pulse_generated == 2: # ref pulses are right, pad end of signal (CHECK IF THIS IS NEEDED)
        #     buffer = [signal[-1]]*9
        #     signal = signal + buffer
        # END HACK

        interpolator = sp.interpolate.interp1d(np.arange(len(signal)), signal)
        interp_signals[i, :] = interpolator(
            np.linspace(0, len(signal) - 1, total_obs)
        )
    ecg_data = pd.DataFrame(
        np.nan,
        index=np.arange(total_obs),
        columns=[lead.name for lead in Format.STANDARD],
    )
    
    # save leftmost pixels for later use
    # these turn out to be the *top* of the reference pulses, usually
    first_pixels = [pulso[0].y for pulso in signal_coords] 
    # Reference pulses
    # the difference between v0 and v1 should be two square grid blocks
    volt_0 = 2*grid_size_px
    volt_1 = 0
    if volt_0 == volt_1:
        raise DigitizationError(
            f"Reference pulses have not been detected correctly"
        )

    for i, lead in enumerate(ORDER):
        zeroed = False
        rhythm = lead in rhythm_leads
        r = rhythm_leads.index(lead) + NROWS if rhythm else i % NROWS
        c = 0 if rhythm else i // NROWS

        # Get correspondent part of the signal for current lead
        signal = interp_signals[r, :]
        obs_num = len(signal) // (1 if rhythm else NCOLS)
        signal = signal[c * obs_num : (c + 1) * obs_num]

        # QA
        for j in range(0, len(interp_signals)):
            if j == r:
                continue
            # Check if this signal overlaps with another, if so, populate with zeros
            shared_points = [point for point in signal if point in interp_signals[j, :]]
            
            # replace y value with first pixel (approx. roi) for this row
            signal_adjusted = np.abs(signal - first_pixels[r])
            if len(shared_points) > len(signal)*0.2: # >20% overlap
                # Only zero if this signal's median y-value is much closer to another line's 
                # first pixel: ie we suspect it's jumped lines
                if np.median(signal - first_pixels[j]) < np.median(signal - first_pixels[r]):
                    signal = np.zeros(len(signal))   
                    zeroed = True          
        
        if not zeroed: 
            # Scale signal with ref pulses
            signal = [(volt_0 - y) * (1 / (volt_0 - volt_1)) for y in signal]
            # use the baseline of the reference pulse, or the first plotted pixel, as the baseline
            # alternate option no longer used: use the median of the signal
            first_pixel_scaled = (volt_0 - first_pixels[r]) * (1 / (volt_0 - volt_1))
            signal = signal - first_pixel_scaled 

        # Round voltages to 4 decimals
        signal = np.round(signal, 4)
        # Cabrera format -aVR
        if ORDER == Format.CABRERA and lead == Lead.aVR:
            signal = -signal
        # Save in correspondent dataframe location
        ecg_data.loc[
            (ecg_data.index >= len(signal) * c)
            & (ecg_data.index < len(signal) * (c + 1)),
            lead.name,
        ] = signal

    return ecg_data, vectorized_signals, grid_size_px