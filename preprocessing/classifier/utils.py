# Adapted from code by Nicola Dinsdale 2020
# Useful functions for training the model
# Args: Class of useful values
# Early stopping: exactly that
# Load pretrained model: loads statedict into model
######################################################################################
import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
import torch
import numpy as np
from sklearn.utils import shuffle
from digitization.Unet.ECGunet import BasicResUNet
from digitization.Unet import patching
from utils import team_helper_code
from tqdm import tqdm
import matplotlib.pyplot as plt

class Args:
    # Store lots of the parameters that we might need to train the model
    def __init__(self):
        self.batch_size = 16
        self.log_interval = 50
        self.learning_rate = 1e-4
        self.epochs = 500
        self.train_val_prop = 0.9 # Set to 1.0 for no validation (train on all data)
        self.patience = 25 # Early stopping patience
        self.channels_first = True
        self.diff_model_flag = False
        self.alpha = 1
        self.ref_dist=None
        self.reduce_lr = True # Decay the learning rate
        self.augmentation = True # Augment the data (rotation, color temp, noise)


class EarlyStopping:
    # Early stops the training if the validation loss doesnt improve after a given patience
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, optimizer, loss, PTH):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, loss, PTH)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print('Early Stopping Counter: ', self.counter, '/', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, loss, PTH)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer, loss, PTH):
        # Saves the model when the validation loss decreases
        if self.verbose:
            print('Validation loss decreased: ', self.val_loss_min, ' --> ',  val_loss, 'Saving model ...')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, PTH + '_checkpoint')
        self.val_loss_min = val_loss


def patch_split_from_ids(ids, generated_patch_dir, real_patch_dir, train_prop, verbose=False, 
                         max_samples=False):
    """
    Shuffles patches and splits them into training and validation sets based on image ID,
    to avoid data leakage. Returns lists of patch filenames for training and validation.

    Args:
    ids: list of image IDs
    im_patch_path (str): path to image patches
    lab_patch_path (str): path to label patches
    train_prop (float in range [0,1]): proportion of patches to use for training
    max_samples (int): maximum number of samples (patches) to use for training and validation
    """
    gen_patch_files = sorted(os.listdir(generated_patch_dir))
    real_pathch_files = sorted(os.listdir(real_patch_dir))
    # make sure there's an associated set of image and label patches for every requested ID
    ids = team_helper_code.check_dirs_for_ids(ids, im_patch_path, lab_patch_path, verbose)

    # there are ~64 patches for each image, shuffle by id to avoid data leakage
    ids = shuffle(list(ids))
    n_images = len(ids)
    split_idx = int(n_images * train_prop)
    img_id_train = ids[:split_idx]
    img_id_test = ids[split_idx:]

    # find all patches for each image id in the train and test sets
    id_train = [f for f in gen_patch_files if f.split('-')[0] in img_id_train]
    if len(img_id_test) > 0: # in case we aren't validating
        id_test = [f for f in gen_patch_files if f.split('-')[0] in img_id_test]
    else:
        id_test = []

    if max_samples and max_samples < len(gen_patch_files):
        n_train_samples = int(max_samples * train_prop)
        n_test_samples = max_samples - n_train_samples
        id_train = shuffle(id_train)[:n_train_samples]
        id_test = shuffle(id_test)[:n_test_samples]

    return id_train, id_test


def save_patches_batch(image_path, label_path, patch_size, patch_save_path, verbose, 
                       delete_images=False):
    """
    Similar to the same function in digitization/Unet/patching.py, but assumes:
    - labels (masks) are PNGs, not numpy arrays
    - images do not necessarily have masks available, but we need to patch them anyway
    """
    im_patch_path = os.path.join(patch_save_path, 'image_patches')
    lab_patch_path = os.path.join(patch_save_path, 'label_patches')
    os.makedirs(im_patch_path, exist_ok=True)
    os.makedirs(lab_patch_path, exist_ok=True)

    im_ids = os.listdir(image_path)
    im_ids = [i.split('.')[0] for i in im_ids]

    # make sure we have matching images and labels
    available_im_ids = team_helper_code.find_available_images(im_ids, image_path, verbose)
    available_label_ids = team_helper_code.find_available_images(im_ids, label_path, verbose)
    ids_with_labels = list(set(available_im_ids).intersection(available_label_ids))

    for id in tqdm(im_ids, desc='Generating and saving patches', disable=not verbose):
        id = id.split('.')[0]
        lab_pth = os.path.join(label_path, id + '.png')
        img_pth = os.path.join(image_path, id + '.png')
        with open(img_pth, 'rb') as f:
            image = plt.imread(f)
        if id in ids_with_labels:
            with open(lab_pth, 'rb') as f:
                label = plt.imread(f)
                # binzarize the label: need False for background, True for signals
                # originally we have background as 255, signals as 0 (hopefully)
                label = (label[:,:,0] < np.median(label[:,:,0])).astype(bool)
                plt.imshow(label)
                plt.show()
        else:
            label = None
        # if image.shape[:-1] != label.shape:
        #     print(f"{id} image shape: {image.shape}, labels shape: {label.shape}")

        im_patches, label_patches = patching.patchify(image, label, size=(patch_size,patch_size))
        
        for i in range(len(im_patches)):
            im_patch = im_patches[i]
            # if im_patch.shape[:-1] != lab_patch.shape:
            #     print(f"Image patch shape: {im_patch.shape}, Label patch shape: {lab_patch.shape}")
            k = f'_{i:03d}'
            np.save(os.path.join(im_patch_path, id + k), im_patch)
            if label is not None:
                lab_patch = label_patches[i]
                np.save(os.path.join(lab_patch_path, id + k), lab_patch)
        
        if delete_images:
            os.remove(img_pth)
            os.remove(lab_pth)