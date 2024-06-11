import os
import numpy as np
import matplotlib.pyplot as plt


def patchify(image, label, size=(128,128)):
    im_arr = []
    lab_arr = []
    i_patches = np.ceil(image.shape[0]/size[0]).astype(int)
    j_patches = np.ceil(image.shape[1]/size[1]).astype(int)
    
    # print(image.shape)
    for i in range(0, i_patches):
        for j in range(0, j_patches):
            im_patch = image[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1], :3] # Remove the alpha channel 
            lab_patch = label[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]] # Patch labels 
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
            im_arr.append(im_patch)
            lab_arr.append(lab_patch)
    im_arr = np.array(im_arr)
    lab_arr = np.array(lab_arr)
    # print(im_arr.shape, flush=True)
    return im_arr, lab_arr


def depatchify_images(patches, size=(256,256), image_shape=(1700, 2200)):
    # extra dimension for channels
    # image = np.zeros((image_shape[0], image_shape[1], 3))
    row_patches = np.ceil(image_shape[0]/size[0]).astype(int)
    col_patches = np.ceil(image_shape[1]/size[1]).astype(int)
    max_width = row_patches*size[0]
    max_height = col_patches*size[1]
    image = np.zeros((max_width, max_height, 3))

    i = 0
    for row in range(0, row_patches):
        for col in range(0, col_patches):
            im_patch = patches[i]
            i += 1
            image[row*size[0]:(row+1)*size[0], col*size[1]:(col+1)*size[1], :] = im_patch
    return image


def depatchify(patches, size=(256,256), image_shape=(1700, 2200)):
    row_patches = np.ceil(image_shape[0]/size[0]).astype(int)
    col_patches = np.ceil(image_shape[1]/size[1]).astype(int)
    max_width = row_patches*size[0]
    max_height = col_patches*size[1]
    image = np.ones((max_width, max_height))

    i = 0
    for row in range(0, row_patches):
        for col in range(0, col_patches):
            if i >= len(patches):
                break
            im_patch = patches[i]
            i += 1
            image[row*size[0]:(row+1)*size[0], col*size[1]:(col+1)*size[1]] = im_patch
    image = image[:image_shape[0], :image_shape[1]]
    return image

