import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import team_helper_code
from PIL import Image
import cv2


def patchify(image, label, size=(128,128)):
    im_arr = []
    lab_arr = []
    i_patches = np.ceil(image.shape[0]/size[0]).astype(int)
    j_patches = np.ceil(image.shape[1]/size[1]).astype(int)

    labels_present = True
    if label is None:
        labels_present = False

    for i in range(0, i_patches):
        for j in range(0, j_patches):
            im_patch = image[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1], :3] # Remove the alpha channel
            if labels_present: 
                lab_patch = label[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]] # Patch labels
            if im_patch.shape[0] != size[0]:
                store = np.ones((size[0],size[1],3))
                store[:im_patch.shape[0], :im_patch.shape[1]] = im_patch
                im_patch = store
                if labels_present:
                    store = np.zeros((size[0],size[1]))
                    store[:lab_patch.shape[0], :lab_patch.shape[1]] = lab_patch
                    lab_patch = store
            if im_patch.shape[1] != size[1]:
                store = np.ones((size[0],size[1],3))
                store[:im_patch.shape[0], :im_patch.shape[1]] = im_patch
                im_patch = store
                if labels_present:
                    store = np.zeros((size[0],size[1]))
                    store[:lab_patch.shape[0], :lab_patch.shape[1]] = lab_patch
                    lab_patch = store
            im_arr.append(im_patch)
            if labels_present:
                lab_arr.append(lab_patch)
    im_arr = np.array(im_arr)
    if labels_present:
        lab_arr = np.array(lab_arr)

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
    image = np.zeros((max_width, max_height))

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
        if label is not None:
            lab_patch = label_patches[i]
            np.save(os.path.join(lab_patch_save_path, k), lab_patch)


def save_patches_batch(ids, image_path, label_path, patch_size, patch_save_path, verbose, 
                       delete_images=True, require_masks=True, max_samples=None):
    im_patch_path = os.path.join(patch_save_path, 'image_patches')
    lab_patch_path = os.path.join(patch_save_path, 'label_patches')
    os.makedirs(im_patch_path, exist_ok=True)
    os.makedirs(lab_patch_path, exist_ok=True)

    # make sure we have matching images and labels
    if require_masks:
        available_im_ids = team_helper_code.find_available_images(ids, image_path, verbose)
        available_label_ids = team_helper_code.find_available_images(ids, label_path, verbose)
        ids = list(set(available_im_ids).intersection(available_label_ids))
    else:
        ids = team_helper_code.find_available_images(ids, image_path, verbose)

    n_images = len(ids)
    save_chance = 1.0
    rng = np.random.default_rng()
    if max_samples is not None: # a hacky way to limit how many patches are saved
        n_patches = n_images * 64 # usually 64 patches per image
        if n_patches > max_samples:
            save_chance = max_samples/n_patches

    for id in tqdm(ids, desc='Generating and saving patches', disable=not verbose):
        id = id.split('.')[0]
        img_pth = os.path.join(image_path, id + '.png')
        with open(img_pth, 'rb') as f:
            image_open = Image.open(f)
            orginal_shape = image_open.size
            if image_open.size[0] > 2300: # index 0 is width
                ratio = 2300/image_open.size[0]
                new_height = int(ratio*image_open.size[1])
                image_open = image_open.resize((2200, new_height), Image.Resampling.LANCZOS)
            image = np.array(image_open)
            

        if os.path.exists(os.path.join(label_path, id + '.npy')):
            lab_pth = os.path.join(label_path, id + '.npy')
            label = np.load(lab_pth, allow_pickle=True)
        elif os.path.exists(os.path.join(label_path, id + '.png')):
            lab_pth = os.path.join(label_path, id + '.png')
            with open(lab_pth, 'rb') as f:
                label_open = Image.open(f)
                if label_open.size != orginal_shape:
                    print("Label shape does not match image shape!")
                if label_open.size != image_open.size:
                    label_open = label_open.resize(image_open.size, Image.Resampling.LANCZOS)
                # binzarize the label: need False for background, True for signals
                # assume if image, we have background as 255, signals as 0 
                label = np.array(label_open)
                blur = cv2.GaussianBlur(label,(5,5),0) # TODO: check if this helps?
                thresh, label = cv2.threshold(label[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                label = label.astype(bool)      
        elif not require_masks:
            label = None

        # if image.shape[:-1] != label.shape:
        #     print(f"{id} image shape: {image.shape}, labels shape: {label.shape}")
        im_patches, label_patches = patchify(image, label, size=(patch_size,patch_size))
        
        for i in range(len(im_patches)):
            im_patch = im_patches[i]
            # if im_patch.shape[:-1] != lab_patch.shape:
            #     print(f"Image patch shape: {im_patch.shape}, Label patch shape: {lab_patch.shape}")
            k = f'-{i:03d}'
            if rng.random() <= save_chance:
                np.save(os.path.join(im_patch_path, id + k), im_patch)
                if label is not None:
                    lab_patch = label_patches[i]
                    np.save(os.path.join(lab_patch_path, id + k), lab_patch)
        
        if delete_images:
            os.remove(img_pth)
            if os.path.exists(lab_pth):
                os.remove(lab_pth)