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
from utils import team_helper_code

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


def load_unet_from_state_dict(state_dict, verbose=False):
    # Load the model
    unet = BasicResUNet(3, 2, nbs=[1, 1, 1, 1], init_channels=16, cbam=False)
    if torch.cuda.is_available():
        unet = unet.cuda()

    encoder_dict = unet.state_dict()
    if verbose:
        print(f'U-net: loaded {len(state_dict)}/{len(unet.state_dict())} weights.')
        if torch.cuda.is_available():
            print('Using cuda.')
    unet.load_state_dict(state_dict)

    return unet