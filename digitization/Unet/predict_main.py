# Adapted from code by Nicola Dinsdale 2024
import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from digitization.Unet.ECGunet import BasicResUNet
from digitization.Unet.datasets.PatchDataset import PatchDataset
from digitization.Unet import entropy_estimator
from digitization import Unet
from utils import team_helper_code
from tqdm import tqdm

import matplotlib.pyplot as plt


def normal_predict(model, test_loader, have_labels=False):
    cuda = torch.cuda.is_available()
    pred = []
    true = []
    orig = []
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            orig.append(data.detach().cpu().numpy())
            if have_labels:
                true.append(target.detach().cpu().numpy())
            x = model(data)
            pred.append(x.detach().cpu().numpy())
    orig = np.array(orig)
    pred = np.array(pred)
    if have_labels:
        true = np.array(true)

    # # randomly save some patch results for comparison
    # results = pred.squeeze()
    # results = np.argmax(results, axis=1)
    # true_patches = np.ones_like(results)
    # orig_patches = orig.squeeze().transpose(0, 2, 3, 1)

    # for patch in range(results.shape[0]):
    #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #     ax[0].imshow(results[patch], cmap='gray')
    #     ax[0].set_title('Predicted Image')
    #     ax[1].imshow(true_patches[patch], cmap='gray')
    #     ax[1].set_title('True Image')
    #     ax[2].imshow(orig_patches[patch], cmap='gray')
    #     ax[2].set_title('Original Image')
    #     # save the plot
    #     results_path = os.path.join("test_data", "patch_results")
    #     plt.savefig(os.path.join(results_path, '0-' + str(patch) + '.png'))

    return pred, orig, true


def calculate_dice(ground_truth, prediction):
    # Calculate the 3D dice coefficient of the ground truth and the prediction
    ground_truth = ground_truth > 0.5  # Binarize volume
    prediction = prediction > 0.5  # Binarize volume
    epsilon = 1e-5  # Small value to prevent dividing by zero
    true_positive = np.sum(np.multiply(ground_truth, prediction))
    false_positive = np.sum(np.multiply(ground_truth == 0, prediction))
    false_negative = np.sum(np.multiply(ground_truth, prediction == 0))
    dice3d_coeff = 2*true_positive / \
        (2*true_positive + false_positive + false_negative + epsilon)
    return dice3d_coeff


def predict_single_image(image_id, im_patch_dir, unet, original_image_size=(1700, 2200)):
    """
    Assumes labels are not present; patches already generated and in im_patch_dir
    MODEL MUST BE PRE-LOADED UNET + STATE DICT
    Returns the predicted image as a numpy array (flat)
    """
    image_id = image_id.split('_')[0]
    patches = os.listdir(im_patch_dir)
    patch_ids = [f for f in patches if f.split('_')[0] == image_id]

    label_patch_dir = os.path.join(im_patch_dir, 'label_patches')
    test_dataset = PatchDataset(patch_ids, im_patch_dir, None, train=False, transform=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    results, _, _ = normal_predict(unet, test_dataloader, have_labels=False)

    results = results.squeeze()
    results = np.argmax(results, axis=1)

    predicted_im = Unet.patching.depatchify(results, results.shape[1:], original_image_size)
    return predicted_im


def batch_predict_full_images(ids_to_predict, patch_dir, unet, save_pth, 
                              verbose, save_all=True):
    """
    Mostly for testing - assumes we have labels for accuracy score and have already generated 
    patches. Note that this may fail if an image is missing one or more patches.
    """
    # want to predict on one image at a time so we can reconstruct it
    im_patch_dir = os.path.join(patch_dir, 'image_patches')
    label_patch_dir = os.path.join(patch_dir, 'label_patches')
    ids = os.listdir(im_patch_dir)
    ids_to_predict = team_helper_code.check_dirs_for_ids(ids_to_predict, im_patch_dir, 
                                                         label_patch_dir, verbose)
    if verbose:
        print(f"Testing on {len(ids_to_predict)} images with {len(ids)} total patches.")

    # Load the model
    dice_list = np.zeros(len(ids_to_predict))
    entropy_list = []

    for i, image_id in tqdm(enumerate(ids_to_predict), desc='Running U-net on images', 
                            disable=not verbose, total=len(ids_to_predict)):
        patch_ids = [f for f in ids if f.split('-')[0] == image_id]
        patch_ids = sorted(patch_ids)
        # train = True here because we want to load the labels for accuracy score
        test_dataset = PatchDataset(patch_ids, im_patch_dir, label_patch_dir, train=True, 
                                    transform=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        results, orig, true = normal_predict(unet, test_dataloader, have_labels=True)
        dice_list[i] = calculate_dice(true, results)
        entropy_list.append(entropy_estimator.entropy_est(results, reduce=True))

        results = results.squeeze()
        results = np.argmax(results, axis=1)

        if save_all:
            save_chance = 1
        else:
            # randomly save some full images
            save_chance = np.random.rand()
        if save_chance > 0.9:
            predicted_im = Unet.patching.depatchify(results, results.shape[1:])
            with open(os.path.join(save_pth, image_id + '.npy'), 'wb') as f:
                np.save(f , predicted_im)

        # # randomly save some patch results for comparison
        # true_patches = np.argmax(true.squeeze(), axis=1)
        # orig_patches = orig.squeeze().transpose(0, 2, 3, 1)

        # for patch in range(results.shape[0]):
        #     save_chance = np.random.rand()
        #     if save_chance > 0.95:
        #         fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        #         ax[0].imshow(results[patch], cmap='gray')
        #         ax[0].set_title('Predicted Image')
        #         ax[1].imshow(true_patches[patch], cmap='gray')
        #         ax[1].set_title('True Image')
        #         ax[2].imshow(orig_patches[patch], cmap='gray')
        #         ax[2].set_title('Original Image')
        #         # save the plot
        #         results_path = os.path.join("test_data", "patch_results")
        #         plt.savefig(os.path.join(results_path, image_id +  '-' + str(patch) + '.png'))

    entropy_list = np.array(entropy_list)
    return dice_list, entropy_list



