# Nicola Dinsdale 2024
# Make ECG training data
#####################################################################################################
# Import dependencies
import numpy as np 
import pickle
import os 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
#####################################################################################################

def patchify(image, label, size=(128,128)):
    im_arr = []
    lab_arr = []
    i_patches = np.ceil(image.shape[0]/size[0]).astype(int)
    j_patches = np.ceil(image.shape[1]/size[1]).astype(int)
    
    print(image.shape)
    for i in range(0, i_patches):
        for j in range(0, j_patches):
            im_patch = image[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1], :3] # Remove the alpha channel 
            lab_patch = label[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]] # Remove the alpha channel 
            if im_patch.shape[0] != size[0]:
                store = np.ones((size[0],size[1],3))
                store[:im_patch.shape[0], :im_patch.shape[1]] = im_patch
                im_patch = store
                store = np.zeros((size[0],size[1]))
                store[:lab_patch.shape[0], :lab_patch.shape[1]] = lab_patch
                lab_patch = store
            if im_patch.shape[1] != size[1]:
                store = np.ones((size[0],size[1],3))
                store[:im_patch.shape[0], :im_patch.shape[1]] = im_patch
                im_patch = store
                store = np.zeros((size[0],size[1]))
                store[:lab_patch.shape[0], :lab_patch.shape[1]] = lab_patch
                lab_patch = store
            if np.sum(lab_patch) > 0:   # Currently only keep patches with some label in 
                im_arr.append(im_patch)
                lab_arr.append(lab_patch)
    im_arr = np.array(im_arr)
    lab_arr = np.array(lab_arr)
    # print(im_arr.shape, flush=True)
    return im_arr, lab_arr

