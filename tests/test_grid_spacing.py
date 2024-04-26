# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:26:38 2024

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

## ---- import image from folder ------
image = 'C:/Users/hssdwo/Desktop/pn2024/00001_lr-0.png'
im = iio.imread(image)
red_im = im[:,:,1]
#red_im[red_im>60]= 255

output_rot = sp.ndimage.rotate(red_im, -1, axes=(1, 0), reshape=True)
fig = plt.figure()
fig.set_figheight(60)
fig.set_figwidth(20)
plt.imshow(output_rot)
a = np.sum(output_rot, axis=0)
plt.figure()
plt.plot(a[0:500])

spec = np.fft.fft(a[500:1500])
plt.figure()
plt.plot(np.abs(spec[3:500]))
freq = np.argmax(np.abs(spec[3:100])) + 3