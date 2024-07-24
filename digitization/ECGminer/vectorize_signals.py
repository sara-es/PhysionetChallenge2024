import numpy as np
import scipy as sp
import pandas as pd
from typing import Iterable, Tuple
from digitization.ECGminer.assets.DigitizationError import DigitizationError
from digitization.ECGminer.assets.Point import Point
from digitization.ECGminer.assets.Format import Format
from digitization.ECGminer.assets.Lead import Lead
from digitization.ECGminer import layout


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
    expected_pulse_width : int,
    is_generated : bool = True,
    scan_perc : float = 0.1,
) -> int:
    """

    """
    baseline = np.median(signal)
    scan_length = round(len(signal) * scan_perc) # look in the first and last 10% of the signal
    # get left hand segment
    left_segment = signal[0:scan_length]
    right_segment = signal[-scan_length:]
    is_left = is_pulse(left_segment, baseline, 
                       min_height=expected_pulse_width-10, 
                       min_width=expected_pulse_width-10)
    is_right = is_pulse(right_segment, baseline, 
                        min_height=expected_pulse_width-10, 
                        min_width=expected_pulse_width-10)

    if is_left & is_right:
        # triggers sometimes when the end of the signal is the same height as the reference pulse
        # should assume it's a mistake, but this causes issues. Instead, assume it's on the left.
        return 1 
    elif is_left:
        return 1 
    elif is_right and not is_generated: # generator never puts ref pulses on the right
        print("RP found on right, but not left.")
        return 2
    else:
        return 0 # will be False if bool


def remove_pulses(raw_signals: Iterable[Iterable[Point]], ref_pulse_present: int,
                  duration: int) -> Tuple[Iterable[Iterable[Point]], Iterable[Tuple[int, int]]]:
    """

    """
    rp_at_right = ref_pulse_present == 2
    INI, MID, END = (0, 1, 2)
    LIMIT = min([len(signal) for signal in raw_signals])
    # scale the maximum amount of pixels to search for reference pulses by duration of the signal
    max_pulse_width = int(max(len(signal) for signal in raw_signals)/(duration/0.2 + 1))
    PIXEL_EPS = 5 # OG ecg-miner constant
    # use first pixel in row as baseline - old version used furthest right, but led to issues
    # TODO should probably use furthest right if ref pulse at right
    furthest_left_pixels = [pulso[0].y for pulso in raw_signals]

    direction = (
        range(-1, -LIMIT, -1) if rp_at_right else range(LIMIT)
    )
    pulse_pos = INI
    ini_count = 0
    cut = None
    for i in direction:
        y_coords = [
            pulso[i].y - ini
            for pulso, ini in zip(raw_signals, furthest_left_pixels)
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
        if abs(i) > max_pulse_width*1.5: # failsafe to make sure we don't clip more than logical
            cut = i # force cut
            break
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
            (furthest_left_pixels[i], ref_pulses[i][-1])
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


def vectorize(signal_coords: Iterable[Iterable[Point]], sig_len: int, max_duration: int,
    is_generated_image: bool = True) -> Tuple[np.array, float, int]:
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
        vectorized_signals: Iterable[Iterable[Point]], x and y coordinates of the signals with
            reference pulses removed.
        gridsize: float, scaling factor for the signals.
    """

    signal_slice = slice(0, None)
    signals = [rs[signal_slice] for rs in signal_coords]
    max_len = max(map(lambda signal: len(signal), signals))
    raw_signals = np.zeros((len(signals), max_len))
    pulses =  np.zeros(len(signals))
    expected_pulse_width = int(max(len(signal) for signal in raw_signals)/(max_duration/0.2 + 1))

    for i in range(len(signals)):
        signal = [-p.y for p in signals[i]] # invert y coords (coords *down* from top)
        # interpolate to linear x coords
        interpolator = sp.interpolate.interp1d(np.arange(len(signal)), signal, fill_value='extrapolate') # current x, y
        raw_signals[i, :] = interpolator(
            np.linspace(0, max_len-1, max_len) # xnew, returns ynew
        ) 
        pulses[i] = (row_pulse_pos(raw_signals[i, :], expected_pulse_width, is_generated_image))

    # if we think we've found reference pulses in 2 or more rows, they're probably present
    if sum(pulses == 1) >= 2:
        ref_pulse_present = 1
    elif sum(pulses == 2) >= 2:
        ref_pulse_present = 2
    else:
        ref_pulse_present = 0

    if ref_pulse_present:
        vectorized_signals, ref_pulses = remove_pulses(signal_coords, ref_pulse_present, max_duration)
    else: # Slice signal
        signal_slice = (
            slice(None, None)
        )
        vectorized_signals = [rs[signal_slice] for rs in signal_coords]

    #### START OF VECTORIZE
    # Now that we've stripped out the reference pulses, calculate grid size and re-interpolate to correct
    # number of samples (this could definitely be done more efficiently)
    max_len = max(map(lambda signal: len(signal), vectorized_signals))
    grid_size_px = max_len / (max_duration/0.2) # 0.2 s per square

    # TODO: check if all rows are the same length, pad if not

    # Force correct sampling frequency/number of samples
    total_obs = int(sig_len)
    # Linear interpolation to get a certain number of observations
    interp_signals = np.empty((len(vectorized_signals), total_obs))
    for i in range(len(vectorized_signals)):
        signal = [p.y for p in vectorized_signals[i]]

        # ATTENTION
        # HACK BELOW
        # This adds a few pixels to the start of the signal, because the image-kit generator
        # overlaps the ecg and the reference pulses slightly. On real images, we should NOT do this!
        if is_generated_image:
            if ref_pulse_present == 1: # ref pulses at left, pad start of signal
                buffer = [signal[0]]*2
                signal = buffer + signal
            if ref_pulse_present == 2: # ref pulses are right, pad end of signal (CHECK IF THIS IS NEEDED)
                buffer = [signal[-1]]*2
                signal = signal + buffer
        # END HACK

        interpolator = sp.interpolate.interp1d(np.arange(len(signal)), signal)
        interp_signals[i, :] = interpolator(
            np.linspace(0, len(signal) - 1, total_obs)
        )

    # Layout detection
    if is_generated_image: # we know the generator only uses certain parameters
        NROWS, NCOLS = (3, 4) 
        is_cabrera = False
    else: 
        NROWS = 3 # ATTENTION: LAYOUT HARD CODED FOR NOW, may have more if multiple rhythm leads 
        NCOLS = 4 # We know Challenge team will only use 3x4 format
        is_cabrera = layout.cabrera_detector(interp_signals, NCOLS)
    ORDER = Format.CABRERA if is_cabrera else Format.STANDARD
    # if is_cabrera:
    #     print('Cabrera format detected!')
    rhythm_leads = layout.detect_rhythm_strip(interp_signals, is_cabrera, THRESH=1)
    # if rhythm_leads != [Lead.II]:
    #     print(f'ALTERNATIVE RHYTHM LEAD DETECTED: {rhythm_leads}')
    
    # Reference pulses
    # save the base of the reference pulses for scaling the signals
    if ref_pulse_present == 2:
        first_pixels = [pulso[-1].y for pulso in signal_coords]
    else:
        first_pixels = [pulso[0].y for pulso in signal_coords]
    
    # the difference between v0 and v1 should be two square grid blocks
    volt_0 = 2*grid_size_px
    volt_1 = 0
    if volt_0 == volt_1:
        raise DigitizationError(
            f"Reference pulses have not been detected correctly"
        )
    
    # Create dataframe and populate with signals
    ecg_data = pd.DataFrame(
        np.nan,
        index=np.arange(total_obs),
        columns=[lead.name for lead in Format.STANDARD],
    )

    for i, lead in enumerate(ORDER):
        zeroed = False
        rhythm = lead in rhythm_leads
        r = rhythm_leads.index(lead) + NROWS if rhythm else i % NROWS
        c = 0 if rhythm else i // NROWS

        # Get correspondent part of the signal for current lead
        signal = interp_signals[r, :]
        full_signal_median = np.median(signal)
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
            signal_other_line_adjusted = np.abs(signal - first_pixels[j])
            if len(shared_points) > len(signal)*0.1: # >10% overlap
                # Check a sliding window of the signal
                # Only zero if this signal's median y-value is much closer to another line's 
                # first pixel: ie we suspect it's jumped lines
                window = int(grid_size_px*2) # two grid squares
                window_medians = np.median(np.lib.stride_tricks.sliding_window_view(signal_adjusted, 
                                                                                (window,)), axis=1)
                window_medians_other_line = np.median(np.lib.stride_tricks.sliding_window_view(
                                                   signal_other_line_adjusted, (window,)), axis=1)
                if any(window_medians_other_line < window_medians):
                    signal = np.zeros(len(signal))   
                    zeroed = True 
                    break
        
        if not zeroed:
            if ref_pulse_present:
                # Scale signal with ref pulses
                signal = [(volt_0 - y) * (1 / (volt_0 - volt_1)) for y in signal]
                # use the baseline of the reference pulse, or the first plotted pixel, as the baseline
                first_pixel_scaled = (volt_0 - first_pixels[r]) * (1 / (volt_0 - volt_1))
                signal = signal - first_pixel_scaled 
            else:
                # use the median of the signal as the baseline
                signal = [(volt_0 - (y)) * (1 / (volt_0 - volt_1)) for y in signal]
                median_scaled = (volt_0 - full_signal_median) * (1 / (volt_0 - volt_1))
                signal = signal - median_scaled
                
        # remove single pixel spikes from lead delimiters in middle of signals
        if is_generated_image:
            pixels_to_cutoff = 6
            if c == 0 and not rhythm: # furthest left column
                signal[-pixels_to_cutoff:] = 0
            elif c == NROWS: # furthest right column
                signal[:pixels_to_cutoff] = 0
            elif not rhythm: # middle columns
                signal[:pixels_to_cutoff] = 0
                signal[-pixels_to_cutoff:] = 0

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