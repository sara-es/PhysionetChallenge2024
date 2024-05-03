"""
Implements a cepstrum-based grid detection algorithm from Dave's test2.py script. Uses Bresenham's line algorithm
to speed up the rotation of the image.
"""
import imageio.v3 as iio
import numpy as np
import scipy as sp
from skimage.morphology import opening
from skimage.draw import line
from image_cleaning import remove_shadow, transforms

import matplotlib.pyplot as plt


# This function takes an input of a raw image in png (RGBA) and outputs the cleaned image.
def clean_image(image, return_modified_image=True):
    im = iio.imread(image)

    # files are png, in RGBa format. The alpha channel is 255 for all pixels (opaque) and therefore totally uniformative.
    im = np.delete(im, np.s_[-1:], axis=2)

    # plot to view the raw image, and the RGB channels individually
    # note: these might be backwards - I think cv2 uses BGR, not RGB
    red_im = im[:, :, 0].astype(np.float32)  # this channel doesn't show up the grid very much
    green_im = im[:,:,1].astype(np.float32)
    blue_im = im[:, :, 2].astype(np.float32)
    im_bw = (0.299*red_im + 0.114*blue_im + 0.587*green_im) # conversion from RGB -> greyscale
    im_bw[im_bw>80] = 255 # magic number to clean image a little bit

    # 2. rotate image
    angle, gridsize = get_rotation_angle(im_bw)

    # 1. remove the shadows and grid
    restored_image = remove_shadow.single_channel_sigmoid(red_im, angle)
    
    # Testing: hack to close up more of the gaps
    restored_image = opening(restored_image, footprint=[(np.ones((3, 1)), 1), (np.ones((1, 3)), 1)])

    return restored_image, gridsize


def get_rotation_angle(greyscale_image):
    # Idea: grid search a bunch of rotations. Find the rotation that gives the most prominent
    # cepstrum peak
    # n.b. ndimage.rotate is super slow - can speed this up a tonne using Bresenham's line algorithm
    # attempt at this: angle for start and end of lines is just going to be y-coord offset. Fix one column 
    # (left one) in the middle and move the other (right col) up and down

    cep_max = []
    cep_idx = []
    min_angle = -10
    max_angle = 10
    max_grid_space = 50

    # cutoff determined by max rotation angle: width of image * tan(max_angle)
    # this breaks if angle is too large
    max_rot = np.max((abs(min_angle), abs(max_angle)))
    cutoff_pixels = int(greyscale_image.shape[1] * np.tan(np.radians(max_rot)))

    # get coords for first (left) column of image, cropped
    # assume top left corner is (1, 1)
    img_height = greyscale_image.shape[0]
    y_coords = np.arange(1, img_height+1)

    left_col_y = y_coords[cutoff_pixels//2:-cutoff_pixels//2]
    left_col_x = np.ones(img_height-cutoff_pixels).astype(np.int32)
    right_col_x = np.ones(img_height-cutoff_pixels).astype(np.int32) * (greyscale_image.shape[1])
    
    cep_max = []
    cep_idx = []

    for i in range(cutoff_pixels):
        # get last (right) column of image: this moves down by one pixel each iteration
        right_col_clip = y_coords[i:-(cutoff_pixels-i)]

        # arrays to store the lines and the image values along the lines
        sklines = np.zeros((left_col_y.shape[0], 2, greyscale_image.shape[1]))
        image_lines = np.zeros((left_col_y.shape[0], greyscale_image.shape[1]))

        for j in range(left_col_y.shape[0]):
            rr, cc = line(left_col_y[j], left_col_x[j], right_col_clip[j], right_col_x[j])
            sklines[j] = np.array([rr, cc])
            image_lines[j] = greyscale_image[rr-1, cc-1]

        col_hist = np.sum(image_lines, axis=1)
        ceps = compute_cepstrum(col_hist)
        ceps = ceps[1:] # remove DC component
        peaks, _ = sp.signal.find_peaks(ceps[1:max_grid_space])
        prominences = sp.signal.peak_prominences(ceps[1:max_grid_space], peaks)
        idx = np.argmax(prominences[0])
        cep_max.append(prominences[0][idx])
        cep_idx.append(peaks[idx])
        
    rot_idx = np.argmax(cep_max)
    print(rot_idx)
    rot_angle = rot_idx #+ min_angle
    grid_length = cep_idx[rot_idx] + 1 #add one to compensate for removing dc component earlier

    return rot_angle, grid_length


def compute_cepstrum(xs):
    cepstrum = np.abs(sp.fft.ifft(np.log(np.absolute(sp.fft.fft(xs)))))
    return cepstrum


def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = sp.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = sp.signal.sosfiltfilt(sos, data)
    return filtered_data


def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = sp.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = sp.signal.sosfiltfilt(sos, data)
    return filtered_data