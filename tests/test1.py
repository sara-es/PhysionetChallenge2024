# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:01:03 2024

@author: hssdwo
"""
import imageio.v3 as iio
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import cv2
from skimage.morphology import closing
from scipy.stats import mode
# For automatic digitisation tools to work, we need to clean up the initial image in stages:
# 1. rotate image (note that this can be applied to image, or to the extracted signal at the end)
# 2. de-shadow image
# 3. identify the 12 lead labels, and segment the image into the 12 leads

## ---- import image from folder ------
im = iio.imread('C:/Users/hssdwo/Desktop/pn2024/00001_hr-0.png')


# files are png, in RGBa format. The alpha channel is 255 for all pixels (opaque) and therefore
# totally uniformative. Let's remove it
im = np.delete(im, np.s_[-1:], axis=2)


# plot to view the raw image, and the RGB channels individually
red_im = im[:,:,0]
green_im = im[:,:,1]
blue_im =  im[:,:,2]

fig, axs = plt.subplots(4)
fig.set_figheight(80)
fig.set_figwidth(20)
axs[0].imshow(im)
axs[1].imshow(red_im)
axs[2].imshow(green_im) 
axs[3].imshow(blue_im)

## ---- process the image to enhance the grid and get rid of shadows
# todo - fix magic number here
dev_im = blue_im - (red_im*1.4) # scaled red image so that it is the same brightness. Hard_coded for now
plt.imshow(dev_im)
dev_im[abs(dev_im)<50] = 0 # enhance image so that any light greys are forced to be white


# ---- get the lines (along with their angles) in the image - we expect many lines with similar angles
# the angle represents the rotation of the image, which we can then undo
copy_im = np.uint8(dev_im)
edges = cv2.Canny(copy_im,50,150,apertureSize = 3)

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

#output of hough transform is array of lines, with 2nd column as angle in radians. I *think* the
# starting angle of zero points west, and this is clockwise rotation.





# --- undo rotation on the gridlines, as proof of principle --- 
angle = 57.29*1.5 # hard coded for the moment - this is degrees/radian * hough transform angle
out = sp.ndimage.rotate(dev_im, -(90-angle), axes=(1, 0), reshape=True)

# morphological filter on red channel? (closing?)
test = closing(red_im, footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)])