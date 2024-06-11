# Adapted from code by Nicola Dinsdale 2020
# Useful functions for training the model
# Args: Class of useful values
# Early stopping: exactly that
# Load pretrained model: loads statedict into model
######################################################################################
import os
import torch
import numpy as np
from sklearn.utils import shuffle

class Args:
    # Store lots of the parameters that we might need to train the model
    def __init__(self):
        self.batch_size = 32
        self.log_interval = 10
        self.learning_rate = 0.5e-3
        self.epochs = 50
        self.train_val_prop = 1.0 # Set to 1.0 for no validation (train on all data)
        self.patience = 25 # Early stopping patience
        self.channels_first = True
        self.diff_model_flag = False
        self.alpha = 1
        self.ref_dist=None
        self.reduce_lr = True # Decay the learning rate
        self.patchsize = 256 # Size of the patches in pixels - assumes square patches
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
            if self .counter >= self.patience:
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
                        'loss': loss}, PTH)


def patch_split_from_ids(ids, im_patch_path, lab_patch_path, train_prop, max_samples=False):
    # TODO: make sure there's an associated label patch for every image patch path
    im_patch_files = sorted(os.listdir(im_patch_path))
    lab_patch_files = sorted(os.listdir(lab_patch_path))
    # there are ~64 patches for each image, shuffle by id to avoid data leakage
    image_ids = set([f.split('_')[0] for f in ids]) # set = unique values
    single_image_patchs = [f for f in im_patch_files if f.split('_')[0] == ids[0].split('_')[0]]
    n_patches_per_image = len(single_image_patchs)
    print(image_ids)
    image_ids = shuffle(list(image_ids), random_state=42)

    if max_samples and max_samples < len(image_ids):
        train_prop = int(max_samples * train_prop)
        img_id_train = image_ids[:train_prop]
        img_id_test = image_ids[train_prop:max_samples]
    else:
        n_images = len(image_ids)
        train_prop = int(n_images * train_prop)
        img_id_train = image_ids[:train_prop]
        img_id_test = image_ids[train_prop:]

    # find all patches for each image id in the train and test sets
    id_train = [f for f in im_patch_files if f.split('_')[0] in img_id_train]
    if len(img_id_test) > 0: # in case we aren't validating
        id_test = [f for f in im_patch_files if f.split('_')[0] in img_id_test]
    else:
        id_test = []

    return id_train, id_test