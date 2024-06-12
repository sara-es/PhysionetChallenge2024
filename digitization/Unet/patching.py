import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def patchify(image, label, size=(128,128)):
    im_arr = []
    lab_arr = []
    i_patches = np.ceil(image.shape[0]/size[0]).astype(int)
    j_patches = np.ceil(image.shape[1]/size[1]).astype(int)
    
    # print(image.shape)
    for i in range(0, i_patches):
        for j in range(0, j_patches):
            im_patch = image[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1], :3] # Remove the alpha channel
            if label: 
                lab_patch = label[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]] # Patch labels
            if im_patch.shape[0] != size[0]:
                store = np.ones((size[0],size[1],3))
                store[:im_patch.shape[0], :im_patch.shape[1]] = im_patch
                im_patch = store
                if label:
                    store = np.zeros((size[0],size[1]))
                    store[:lab_patch.shape[0], :lab_patch.shape[1]] = lab_patch
                    lab_patch = store
            if im_patch.shape[1] != size[1]:
                store = np.ones((size[0],size[1],3))
                store[:im_patch.shape[0], :im_patch.shape[1]] = im_patch
                im_patch = store
                if label:
                    store = np.zeros((size[0],size[1]))
                    store[:lab_patch.shape[0], :lab_patch.shape[1]] = lab_patch
                    lab_patch = store
            im_arr.append(im_patch)
            if label:
                lab_arr.append(lab_patch)
    im_arr = np.array(im_arr)
    if label:
        lab_arr = np.array(lab_arr)
    # print(im_arr.shape, flush=True)
    return im_arr, lab_arr


def depatchify_images(patches, size=(256,256), image_shape=(1700, 2200)):
    """
    Depatchify the patches of images back into the original image shape
    Meant for images with 3 channels (3D/last channel is RGB)
    """
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
    """
    Depatchify the patches of predicted labels back into the original image shape
    Meant for numpy arrays (2D/flat)
    """
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


def save_patches_single_image(record_id, image, label, patch_size, im_patch_save_path, lab_patch_save_path):
    im_patches, label_patches = patchify(image, label, size=(patch_size,patch_size))
    for i in range(len(im_patches)):
        im_patch = im_patches[i]
        k = f'{record_id}_{i:03d}'
        np.save(os.path.join(im_patch_save_path, k), im_patch)
        if label:
            lab_patch = label_patches[i]
            np.save(os.path.join(lab_patch_save_path, k), lab_patch)


def save_patches_batch(image_path, label_path, patch_size, patch_save_path, verbose, 
                       max_samples=False):
    ids = sorted(os.listdir(label_path))
    if max_samples:
        ids = ids[:max_samples]
    im_patch_path = os.path.join(patch_save_path, 'image_patches')
    lab_patch_path = os.path.join(patch_save_path, 'label_patches')
    os.makedirs(im_patch_path, exist_ok=True)
    os.makedirs(lab_patch_path, exist_ok=True)

    for id in tqdm(ids, desc='Generating and saving patches', disable=~verbose):
        lab_pth = os.path.join(label_path, id)
        id = id.split('.')[0]
        img_pth = os.path.join(image_path, id + '.png')

        image = plt.imread(img_pth)
        with open(lab_pth, 'rb') as f:
            label = np.load(f)

        im_patches, label_patches = patchify(image, label, size=(patch_size,patch_size))
        
        for i in range(len(im_patches)):
            im_patch = im_patches[i]
            lab_patch = label_patches[i]
            k = f'_{i:03d}'
            np.save(os.path.join(im_patch_path, id + k), im_patch)
            np.save(os.path.join(lab_patch_path, id + k), lab_patch)
        # np.save(os.path.join(im_patch_path, id), im_patches)
        # np.save(os.path.join(lab_patch_path, id), label_patches)