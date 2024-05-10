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
        # breakpoint()
        
        # get height and index of the most prominent cepstrum peak
        peaks, _ = sp.signal.find_peaks(ceps[1:max_grid_space])
        prominences = sp.signal.peak_prominences(ceps[1:max_grid_space], peaks)
        idx = np.argmax(prominences[0])
        # ratio of peak height to total power in spectrum
        peak_energy = prominences[0][idx] / np.sqrt(np.sum(ceps[1:max_grid_space]**2))
        cep_max.append(peak_energy)
        cep_idx.append(peaks[idx])

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(ceps[1:max_grid_space]) 
        # plt.title('energy:' + str(peak_energy))
        # plt.show()
        # plt.close()
        
    rot_idx = np.argmax(cep_max)
    rot_angle = rot_idx + min_angle
    grid_length = cep_idx[rot_idx] + 1 #add one to compensate for removing dc component earlier

    # If the image_width:grid ratio is too small, we've picked up the small grid
    # On generated images, this ratio is reliably about 60 or 350+
    # so 150 is rather arbitrary but should be fine
    if greyscale_image.shape[1]/grid_length > 150: 
        print('small grid detected ' + str(greyscale_image.shape[1]/grid_length))
        grid_length *= 6
        rot_idx = np.argsort(cep_max)[-2] # second highest peak
        rot_angle = rot_idx + min_angle

    # sub-pixel interpolation to further refine gridsize
    rot_image = sp.ndimage.rotate(greyscale_image, rot_angle, axes=(1, 0), reshape=True)
    col_hist = np.sum(rot_image, axis=0)
    # 1.) find initial peak - start 1/5 way through to avoid edge effects
    init_idx = np.argmax(col_hist[len(col_hist)//5:len(col_hist)//5+grid_length]) + len(col_hist)//5 # very low chance of ties
    x = init_idx
    steps = 10
    search_pix_radius = 2

    # 2.) repeat for 10:
    for i in range(steps):
        x += grid_length
        y = col_hist[x-search_pix_radius: x+search_pix_radius]
        # check left and right to see if there's a higher peak
        x = x + np.argmin(y) - search_pix_radius

    grid_length_adj = (x-init_idx)/steps
    print(f' grid length {grid_length} adjusted: {grid_length_adj}')

    return rot_angle, grid_length_adj


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