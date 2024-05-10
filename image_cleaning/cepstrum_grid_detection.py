"""
Implements a cepstrum-based grid detection algorithm from Dave's test2.py script.
"""
import imageio.v3 as iio
import numpy as np
import scipy as sp
from skimage.morphology import opening
from image_cleaning import remove_shadow


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
    # We assume that grid spacing is under 50 pixels

    cep_max = []
    cep_idx = []
    min_angle = -40
    max_angle = 40
    max_grid_space = 50 # CAREFUL, hard coded for now

    box_width = greyscale_image.shape[1]//3

    for angle in range(min_angle, max_angle): 
        rot_image = sp.ndimage.rotate(greyscale_image, angle, axes=(1, 0), reshape=True)
        # check only the centre of the image, then
        # sum each row. It shouldn't matter if this is rows or columns... but it does
        col_hist = np.sum(rot_image[box_width:2*box_width, box_width:2*box_width], axis = 1) 
        
        ceps = compute_cepstrum(col_hist)
        ceps = ceps[1:] # remove DC component
        
        # get height and index of the most prominent cepstrum peak
        # plt.figure()
        # plt.plot(ceps[1:max_grid_space]) 
        peaks, _ = sp.signal.find_peaks(ceps[1:max_grid_space])
        prominences = sp.signal.peak_prominences(ceps[1:max_grid_space], peaks)
        idx = np.argmax(prominences[0])
        cep_max.append(prominences[0][idx])
        cep_idx.append(peaks[idx])
        
    rot_idx = np.argmax(cep_max)
    rot_angle = rot_idx + min_angle
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