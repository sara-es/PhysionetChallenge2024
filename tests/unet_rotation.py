# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:07:47 2024

hack at rotation script

@author: hssdwo
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# image_folder = "C:/Users/hssdwo/Documents/GitHub/PhysionetChallenge2024/temp_data/masks"

# files = os.listdir(image_folder)
# files = files[0]

# full_path = image_folder + '/' + files

# test_im = np.load(full_path)

image_folder = "C:/Users/hssdwo/Documents/GitHub/PhysionetChallenge2024/temp_data/masks"

files = os.listdir(image_folder)
files = [files[0]]

for file in files:
    full_path = image_folder + '/' + file

    test_im = np.load(full_path)
    angle = 6
    test_im = sp.ndimage.rotate(test_im, angle, axes=(1, 0), reshape=True)

    # ------------------------------------ Rotation testing --------------------------------------------------------#
    # idea: assumes that the active region, ie. that contains any ecg, is minimised in the vertical direction at the best rotation
    # NOTE - this currently assumes that the image is between -45 and 45.
    #--------------------------------------------------------------------------------------------------------------
    
    min_angle = -10
    max_angle = 10
    
    # initialise active region and rotation angle. rot_angle set absurdly high for testing only
    # TODO: set rot_angle back to zero for default
    
    active = np.shape(test_im)[1] # set to image width
    rot_angle = 1000
    
    for angle in range(min_angle, max_angle): # for debugging, this is only searching -5 to +4 degrees  
        rot_image = sp.ndimage.rotate(test_im, angle, axes=(1, 0), reshape=True)
        plt.plot(rot_image)
        col_hist = np.sum(rot_image, axis = 0) #sum each column
        # find the starting and end column - columns with black pixels within the active region
        idxs = np.where(col_hist > 0)[0]
        #startcol = idxs[0]
        #endcol = idxs[-1]
        this_active = len(idxs)
        print(this_active)

        if this_active < active:
            active = this_active
            rot_angle = angle
  
    print(rot_angle)
