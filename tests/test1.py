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
from sklearn.neighbors import KernelDensity
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
scale = np.mean(blue_im)/np.mean(red_im)
dev_im = blue_im - (red_im*scale) # scaled red image so that it is the same brightness. Hard_coded for now
plt.figure()
plt.imshow(dev_im)
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

# morphological filter on red channel? (closing?)
test = closing(red_im, footprint=[(np.ones((7, 1)), 1), (np.ones((1, 7)), 1)])

output_im = red_im - test
plt.figure()
plt.imshow(output_im)

# darken the 9.2% darkest pixels
vals = output_im.flatten()
b = np.histogram(vals,255)
a = np.cumsum(b[0])
thresh = np.argmax(a[a<345260])

output_im[output_im<thresh] = 0 # need to figure out this magic number
plt.figure()
plt.imshow(output_im)
output_rot = sp.ndimage.rotate(output_im, rot_angle, axes=(1, 0), reshape=True)
plt.figure()
plt.imshow(output_rot)

# # ------ Work out the size of the ECG grid (in pixels) by finding distance between grid lines
# # for the moment: --------------------------------------------------------------------------
# #   assume that aspect ratio is always 1:1
# #   assume that we can get this from the hough lines (hough lines are a bit flaky, so might need another way)

# # find the biggest peak
# # find the mode of the biggest peak
# offsets = lines[:,0,0]
# gaps = np.diff(offsets)
# density = sp.stats.gaussian_kde(gaps, bw_method=0.01)
# gap_hist = density(list(range(100)))
# ave_gap = np.argmax(gap_hist)

# # now multiply to get a whole 2.5 secs of ECG
# block_width = 12.5*ave_gap
# block_height = 7 *ave_gap

# # find the start of the top-left ECG, and place the grids (this is probably fragile). A better alternative
# # might be to find the writing, and then work out where the corresponding ECG channel is


# send the 12 ECG segments to the digitizer