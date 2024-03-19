# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:51:53 2024

@author: hssdwo
"""
import imageio.v3 as iio
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import cv2
from skimage.morphology import closing
from scipy.stats import mode

def digitize(image):
    cleaned_image = clean_image(image)
    ECG_signals = digitize_image(cleaned_image)
    return(ECG_signals)

# This function takes an input of a raw image in png (RGBA) and outputs the cleaned image.
def clean_image(image):
    im = iio.imread(image)

    # files are png, in RGBa format. The alpha channel is 255 for all pixels (opaque) and therefore totally uniformative.
    im = np.delete(im, np.s_[-1:], axis=2)


    # plot to view the raw image, and the RGB channels individually
    red_im = im[:,:,0] # this channel doesn't show up the grid very much
    blue_im =  im[:,:,2]
    
    # 2. rotate image
    angle = get_rotation_angle(blue_im, red_im)
    
    # 1. remove the shadows and grid
    restored_image = remove_shadow(red_im, angle)
    
    return restored_image

def digitize_image(restored_image):
    ECG_signals = []
    return ECG_signals

# Simple function to remove shadows - room for much improvement.
def remove_shadow(red_im, angle):
    
    output_im = close_filter(red_im, 9) # this removes the shadows
    output_im0 = close_filter(red_im, 2) # this removes the filter artifacts

    sigmoid_norm1 = 255*sigmoid(norm_rescale(output_im-0.95*output_im0, contrast=8))
    sigmoid_std1 = 255*sigmoid(std_rescale(output_im-0.95*output_im0, contrast=8))

    # feel like we can combine these somehow to be useful?
    combo1 = -(sigmoid_norm1 - sigmoid_std1) # I have no idea why this works, but it does

    greyscale_out = 255*zero_one_rescale(sp.ndimage.rotate(combo1, angle, axes=(1, 0), reshape=True, cval=combo1.mean()))
    cleaned_image = sigmoid_gen(greyscale_out,10,100)
    cleaned_image = 255*zero_one_rescale(cleaned_image)
    return cleaned_image

# returns the rotation angle, assuming an angle of up to +/- 45 degrees. Tested on 10 images so far.
def get_rotation_angle(blue_im, red_im):
    ## process image to enhance the grid and get rid of shadows
    scale = np.mean(blue_im)/np.mean(red_im)
    dev_im = blue_im - (red_im*scale) # scaled red image so that it is the same brightness. Hard_coded for now

    #scale so that white is 255, and black is 0
    im_range = np.ptp(dev_im)
    im_min = np.min(dev_im)
    dev_im = (dev_im - im_min)*255/im_range

    dev_im[abs(dev_im)<50] = 0 # enhance image so that any light greys are forced to be white

    # ---- get the lines (along with their angles) in the image - we expect many lines with similar angles
    # the angle represents the rotation of the image, which we can then undo
    copy_im = np.uint8(dev_im)
    edges = cv2.Canny(copy_im,50,150,apertureSize = 3)


    #output of hough transform is array of lines, with 2nd column as angle in radians. I *think* the
    # starting angle of zero points west, and this is clockwise rotation.
    lines = cv2.HoughLines(edges,1,np.pi/180,800)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(dev_im,(x1,y1),(x2,y2),(0,0,255),2)

    # extract the angles of the lines - if all is well, we should find two main angles corresponding
    # to the horizontal and vertical lines
    angles = lines[:,0,1]*180/np.pi #angles in degrees
    angles = angles[angles<90]
    rot_angle = mode(angles, keepdims = False)
    if rot_angle[0]>45:
        rot_angle = rot_angle[0] - 90
    else:
        rot_angle = rot_angle[0]
    return rot_angle

def close_filter(image, footprint):
    # morphological filter on red channel? (closing?)
    test = closing(image, footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)])
    output_im = image - test
    return output_im

def std_rescale(image, contrast=1):
    """
    Standardizes to between [-contrast, contrast] regardless of range of input
    """
    ptp_ratio = contrast/np.ptp(image)
    shift = (np.max(image) + np.min(image))/contrast # works with negative values
    return ptp_ratio*(image - shift)

def norm_rescale(image, contrast=1):
    """
    Normalizes to zero mean and scales to have one of (abs(max), abs(min)) = contrast
    """
    scale = np.max(np.abs([np.max(image), np.min(image)]))
    return contrast*(image - np.mean(image))/scale

def zero_one_rescale(image):
    """
    Rescales to between 0 and 1
    """
    return (image - np.min(image))/np.ptp(image)

def sigmoid(x):
    """
    tanh sigmoid function
    """
    return 1./(1. + np.exp(-x))

def sigmoid_gen(x, k, x_0):
    """
    tanh sigmoid function
    """
    return 1./(1. + np.exp(-k*(x - x_0)))

