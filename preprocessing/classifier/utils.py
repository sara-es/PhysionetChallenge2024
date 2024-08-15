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


def patch_split_classification(generated_patch_dir, real_patch_dir, train_prop, verbose=False, 
                               max_samples=False):
    """
    Shuffles patches and splits them into training and validation sets based on image ID,
    to avoid data leakage. Returns lists of patch filenames for training and validation.

    Args:
    im_patch_path (str): path to image patches
    lab_patch_path (str): path to label patches
    train_prop (float in range [0,1]): proportion of patches to use for training
    max_samples (int): maximum number of samples (patches) to use for training and validation
    """
    gen_patch_files = sorted(team_helper_code.find_files(generated_patch_dir, '.png'))
    real_patch_files = sorted(team_helper_code.find_files(real_patch_dir, '.png'))
    print(f"Found {len(gen_patch_files)} generated patches and {len(real_patch_files)} real patches.")
    print(f"{real_patch_files}")

    # there are ~64 patches for each image, shuffle by id to avoid data leakage
    real_img_ids = list(set([f.split('_')[0] for f in real_patch_files]))
    real_img_ids = shuffle(real_img_ids)
    n_real_images = len(real_img_ids)
    split_idx = int(n_real_images * train_prop)
    real_img_id_train = real_img_ids[:split_idx]
    real_img_id_test = real_img_ids[split_idx:]

    # find all patches for each real image id in the train and test sets
    real_id_train = [f for f in real_patch_files if f.split('_')[0] in real_img_id_train]
    real_id_test = [f for f in real_patch_files if f.split('_')[0] in real_img_id_test]






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


def prepare_classifier_data_arrays(real_images_folder, gen_patch_folder, patch_size, train_val_prop, verbose, 
                       delete_images=False, data_save_path=None):
    """
    Similar to the same function in digitization/Unet/patching.py, but assumes:
    - labels (masks) are PNGs, not numpy arrays, with black signals on white background
    - images do not necessarily have masks available, but we need to patch them anyway
    - save all patches in one big numpy array, not separate files
    """
    if verbose:
        print("Finding classifier data...", flush=True)

    # split real images into training and validation sets based on image ID
    r_im_ids = os.listdir(real_images_folder)
    r_im_ids = [i.split('.')[0] for i in r_im_ids]
    r_train_ids = shuffle(r_im_ids)[:int(len(r_im_ids)*train_val_prop)]
    r_test_ids = [i for i in r_im_ids if i not in r_train_ids]

    # get filepaths for generated patches, split into training and val sets
    g_patches = os.listdir(gen_patch_folder)
    g_im_ids = shuffle(list(set([f.split('-')[0] for f in g_patches])))
    g_train_im_ids = g_im_ids[:int(len(g_im_ids)*train_val_prop)]
    g_test_im_ids = g_im_ids[int(len(g_im_ids)*train_val_prop):]

    # find all patches for each generated image id in the train and test sets
    g_train_ids = [f for f in g_patches if f.split('-')[0] in g_train_im_ids]
    g_test_ids = [f for f in g_patches if f.split('-')[0] in g_test_im_ids]

    if verbose:
        print("Generating classifier patches...", flush=True)

    n = 8306 * 2 # this is how many patches we expect to generate
    with tqdm(total=int(n)) as pbar:
        for i, id in enumerate(r_train_ids):
            img_pth = os.path.join(real_images_folder, id + '.png')
            with open(img_pth, 'rb') as f:
                image = plt.imread(f)
            im_patches, _ = patching.patchify(image, None, size=(patch_size,patch_size))
            if i == 0:
                X_train = im_patches
                y_train = [1] * len(im_patches)
            else:
                X_train = np.append(X_train, im_patches, axis=0)
                y_train.extend([1] * len(im_patches))
            print(f"{id} patches: {len(im_patches)}")
            pbar.update(len(im_patches))

        for i, id in enumerate(r_test_ids):
            img_pth = os.path.join(real_images_folder, id + '.png')
            with open(img_pth, 'rb') as f:
                image = plt.imread(f)
            im_patches, _ = patching.patchify(image, None, size=(patch_size,patch_size))
            if i == 0:
                X_test = im_patches
                y_test = [1] * len(im_patches)
            else:
                X_test = np.append(X_test, im_patches, axis=0)
                y_test.extend([1] * len(im_patches))
            pbar.update(len(im_patches))

        print(f"Total real patches: {len(y_train)}")
        # equalize ratio of real to generated patches in training set
        g_train_ids = shuffle(g_train_ids)[:len(y_train)]
        g_test_ids = shuffle(g_test_ids)[:len(y_test)]

        for id in g_train_ids:
            img_pth = os.path.join(gen_patch_folder, id)
            im_patch = np.load(img_pth) # already patchified
            X_train = np.append(X_train, im_patch)
            y_train.append(0)
            pbar.update(1)
            
        for id in g_test_ids:
            img_pth = os.path.join(gen_patch_folder, id)
            im_patch = np.load(img_pth)
            X_test = np.append(X_test, im_patch)
            y_test.append(0)
            pbar.update(1)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if data_save_path is not None:
        np.save(data_save_path + 'X_train_' + str(patch_size), X_train)
        np.save(data_save_path + 'y_train_' + str(patch_size), y_train)
        np.save(data_save_path + 'X_test_' + str(patch_size), X_test)
        np.save(data_save_path + 'y_test_' + str(patch_size), y_test)

    return X_train, y_train, X_test, y_test


def prepare_classifier_data(real_patch_folder, gen_patch_folder, train_val_prop, verbose, 
                       delete_images=False, data_save_path=None):
    """
    Splits patches into training and validation sets based on image ID, to avoid data leakage.
    Returns lists of patch filePATHS for training and validation, and their corresponding labels.
    """
    if verbose:
        print("Finding classifier data...", flush=True)

    # split real patches into training and validation sets based on image ID
    r_patch_ids = os.listdir(real_patch_folder)
    r_im_ids = list(set([i.split('_')[0] for i in r_patch_ids]))
    r_train_ids = shuffle(r_im_ids)[:int(len(r_im_ids)*train_val_prop)]
    r_test_ids = [i for i in r_im_ids if i not in r_train_ids]

    # find all patches for each real image id in the train and test sets
    r_train_paths = [os.path.join(real_patch_folder, f) for f in r_patch_ids 
                     if f.split('_')[0] in r_train_ids]
    r_test_paths = [os.path.join(real_patch_folder, f) for f in r_patch_ids 
                  if f.split('_')[0] in r_test_ids]

    # get filepaths for generated patches, split into training and val sets
    g_patches = os.listdir(gen_patch_folder)
    g_im_ids = shuffle(list(set([f.split('-')[0] for f in g_patches])))
    g_train_im_ids = g_im_ids[:int(len(g_im_ids)*train_val_prop)]
    g_test_im_ids = g_im_ids[int(len(g_im_ids)*train_val_prop):]

    # find all patches for each generated image id in the train and test sets
    g_train_paths = [os.path.join(gen_patch_folder, f) for f in g_patches 
                   if f.split('-')[0] in g_train_im_ids]
    g_test_paths = [os.path.join(gen_patch_folder, f) for f in g_patches 
                  if f.split('-')[0] in g_test_im_ids]

    X_train = r_train_paths + g_train_paths
    y_train = np.array([1] * len(r_train_paths) + [0] * len(g_train_paths))
    X_test = r_test_paths + g_test_paths
    y_test = np.array([1] * len(r_test_paths) + [0] * len(g_test_paths))

    return X_train, y_train, X_test, y_test