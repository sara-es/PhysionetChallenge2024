import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np

import helper_code


def get_signal_length_in_pixels(record, mask):
    """
    Returns the length of the signal in pixels.
    """
    # reconstruct signals from u-net outputs
    header_txt = helper_code.load_header(record)
    signal_length = helper_code.get_num_samples(header_txt)
    

if __name__=="__main__":
    mask_path = "test_rot_data\\unet_outputs\\00391_hr.npy"
    mask = np.load(mask_path)
    record = "test_rot_data\\test_images\\00391_hr"
    print("Signal length in pixels: ", get_signal_length_in_pixels(record, mask))
    