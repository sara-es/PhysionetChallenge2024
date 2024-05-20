# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:51:53 2024

@author: hssdwo
"""
import imageio.v3 as iio
import numpy as np
import scipy as sp
import cv2
from skimage.morphology import opening
from scipy.stats import mode
from image_cleaning import remove_shadow


# This function takes an input of a raw image in png (RGBA) and outputs the cleaned image.
def clean_image(image, return_modified_image=True):
    im = iio.imread(image)

    # files are png, in RGBa format. The alpha channel is 255 for all pixels (opaque) and therefore totally uniformative.
    im = np.delete(im, np.s_[-1:], axis=2)

    # plot to view the raw image, and the RGB channels individually
    # note: these might be backwards - I think cv2 uses BGR, not RGB
    red_im = im[:, :, 0].astype(np.float32)  # this channel doesn't show up the grid very much
    blue_im = im[:, :, 2].astype(np.float32)

    # 2. rotate image
    angle, gridsize = get_rotation_angle(blue_im, red_im)

    # 1. remove the shadows and grid
    restored_image = remove_shadow.single_channel_sigmoid(red_im, angle)
    
    # Testing: hack to close up more of the gaps
    restored_image = opening(restored_image, footprint=[(np.ones((3, 1)), 1), (np.ones((1, 3)), 1)])

    return restored_image, gridsize


# returns the rotation angle, assuming an angle of up to +/- 45 degrees. Tested on 10 images so far.
def get_rotation_angle(blue_im, red_im):
    ## process image to enhance the grid and get rid of shadows
    scale = np.mean(blue_im) / np.mean(red_im)
    dev_im = blue_im - (red_im * scale)  # scaled red image so that it is the same brightness. Hard_coded for now

    # scale so that white is 255, and black is 0
    im_range = np.ptp(dev_im)
    im_min = np.min(dev_im)
    dev_im = (dev_im - im_min) * 255 / im_range

    dev_im[abs(dev_im) < 50] = 0  # enhance image so that any light greys are forced to be white

    # ---- get the lines (along with their angles) in the image - we expect many lines with similar angles
    # the angle represents the rotation of the image, which we can then undo
    copy_im = np.uint8(dev_im)
    edges = cv2.Canny(copy_im, 50, 150, apertureSize=3)

    # output of hough transform is array of lines, with 2nd column as angle in radians. I *think* the
    # starting angle of zero points west, and this is clockwise rotation.
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 800)
    # extract the angles of the lines - if all is well, we should find two main angles corresponding
    # to the horizontal and vertical lines
    angles = lines[:, 0, 1] * 180 / np.pi  # angles in degrees

    # main rotation angles are going to be multiples of 90 degrees
    rot_angle = mode((angles%90).astype(int), keepdims=False)
    if rot_angle[0] > 45:
        rot_angle = rot_angle[0] - 90
    else:
        rot_angle = rot_angle[0]
        
    # ------ Work out the size of the ECG grid (in pixels) by finding distance between grid lines
    # for the moment: --------------------------------------------------------------------------
    #   assume that aspect ratio is always 1:1
    #   assume that we can get this from the hough lines (hough lines are a bit flaky, so might need another way)
    offsets = lines[:,0,0]
    gaps = np.diff(np.sort(offsets)) # sort the offsets first in increasing order -> gaps are positive
    density = sp.stats.gaussian_kde(gaps, bw_method=0.01)
    
    gap_hist = density(list(range(100)))
    # instead of taking biggest peak, we can set a minimum gap size in pixels
    min_gap = 30
    gap_peaks = np.argsort(gap_hist)
    ave_gap = gap_peaks[gap_peaks > min_gap][-1]
    # fallback to stop reference pulse error in case gridline detection fails
    if ave_gap == 0:
        ave_gap = 1
    return rot_angle, ave_gap



