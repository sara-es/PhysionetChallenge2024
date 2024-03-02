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

# This function takes an input of a raw image in png (RGBA). It removes image artefacts and outputs the cleaned image.
# Specific artefacts removed are:
# - gridlines
# - rotations
# - shadows
def clean_image(image):
    im = iio.imread(image)

    # files are png, in RGBa format. The alpha channel is 255 for all pixels (opaque) and therefore totally uniformative.
    im = np.delete(im, np.s_[-1:], axis=2)


    # plot to view the raw image, and the RGB channels individually
    red_im = im[:,:,0] # this channel doesn't show up the grid very much
    green_im = im[:,:,1]
    blue_im =  im[:,:,2]
    
    # remove the shadows
    restored_image = remove_shadow(red_im)
    
    # rotate image
    angle = get_rotation_angle(blue_im, red_im)
    output_im = red_im - restored_image
    output_im[output_im<150] = 0
    output_im = sp.ndimage.rotate(output_im, angle, axes=(1, 0), reshape=True)
    return(output_im)
    

# Simple function to remove shadows - room for much improvement.
def remove_shadow(red_im):
    # morphological filter on red channel? (closing?)
    test = closing(red_im, footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)])
    output_im = red_im - test
    
    # n.b. line below doesn't make sense. Need to scale image into 0-255 first
    output_im[output_im<150] = 0 #make blacks blacker. The proper way to do this is to squeeze the pixels through a logistic-like curve
    return output_im[0]

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



