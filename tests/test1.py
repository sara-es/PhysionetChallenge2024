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

## ---- import image from folder ------
im = iio.imread('C:/Users/hssdwo/Desktop/pn2024/00001_hr-0.png')
# remove last column
im = np.delete(im, np.s_[-1:], axis=2)

plt.imshow(im)
plt.show()

red_im = im[:,:,0]
plt.imshow(red_im) # this does not show the grids

green_im = im[:,:,1]
plt.imshow(green_im)

blue_im =  im[:,:,2]
plt.imshow(blue_im) # this shows the grids well


## ---- process the image to enhance the grid and get rid of shadows

dev_im = blue_im - (red_im*1.4) # scaled red image so that it is the same brightness. Hard_coded for now
plt.imshow(dev_im)
dev_im[abs(dev_im)<50] = 0


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

#output of hough transform is array of lines, with 2nd column as angle in radians. I *think* the
# starting angle of zero points west, and this is clockwise rotation.


# --- undo rotation on the gridlines, as proof of principle --- 
angle = 57.29*1.5 # hard coded for the moment - this is degrees/radian * hough transform angle
out = sp.ndimage.rotate(dev_im, -(90-angle), axes=(1, 0), reshape=True)

