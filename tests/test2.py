# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:33:35 2024

test scripts to extend existing methods to monochrome images

@author: hssdwo
"""

import os
import imageio.v3 as iio
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import cv2
from skimage.morphology import closing
from scipy.stats import mode
from sklearn.neighbors import KernelDensity
# For automatic digitisation tools to work, we need to clean up the initial image in stages:
# 1. rotate image (note that this can be applied to image, or to the extracted signal at the end)
# 2. de-shadow image
# 3. identify the 12 lead labels, and segment the image into the 12 leads

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

## ---- import image from folder ------
image = 'C:/Users/hssdwo/Desktop/pn2024/00002_hr-0.png'
im = iio.imread(image)
#im = iio.imread(os.path.join('tiny_testset', 'records500', '00001_hr-0.png'))

# # plot to view the raw image, and the RGB channels individually
blue_im = im[:,:,0]
green_im = im[:,:,1]
red_im =  im[:,:,2]
im_bw = (0.299*red_im + 0.114*blue_im + 0.587*green_im) # conversion from RGB -> greyscale
im_bw[im_bw>80] = 255 # magic number to clean image a little bit

# Plot image for debugging
#plt.imshow(im_bw)
#plt.show()


# Idea: grid search a bunch of rotations. Find the rotation that gives the most prominent
# cepstrum peak
# n.b. ndimage.rotate is super slow - can speed this up a tonne using Bresenham's line algorithm
# We assume that grid spacing is under 50 pixels

cep_max = []
cep_idx = []
min_angle = -5
max_angle = 5
max_grid_space = 50
for angle in range(min_angle, max_angle): # for debugging, this is only searching -5 to +4 degrees
    rot_image = sp.ndimage.rotate(im_bw, angle, axes=(1, 0), reshape=True)
    col_hist = np.sum(rot_image, axis = 1) #sum each row. It shouldn't matter if this is rows or columns... but it does
    
    ceps = compute_cepstrum(col_hist)
    ceps = ceps[1:] # remove DC component
    
    # get height and index of the most prominent cepstrum peak
    plt.figure()
    plt.plot(ceps[1:max_grid_space]) 
    peaks, _ = sp.signal.find_peaks(ceps[1:max_grid_space])
    prominences = sp.signal.peak_prominences(ceps[1:max_grid_space], peaks)
    idx = np.argmax(prominences[0])
    cep_max.append(prominences[0][idx])
    cep_idx.append(peaks[idx])
    
rot_idx = np.argmax(cep_max)
rot_angle = rot_idx + min_angle
grid_length = cep_idx[rot_idx] + 1 #add one to compensate for removing dc component earlier

