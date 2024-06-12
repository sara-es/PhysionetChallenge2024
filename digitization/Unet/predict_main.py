# Adapted from code by Nicola Dinsdale 2024
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from digitization.Unet.datasets.PatchDataset import PatchDataset
from digitization.Unet import patching, utils
from tqdm import tqdm

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

    return pred, orig, true


def dice(ground_truth, prediction):
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


def predict_single_image(image_id, im_patch_dir, unet):
    """
    Assumes labels are not present; patches already generated and in im_patch_dir
    MODEL MUST BE PRE-LOADED UNET + STATE DICT
    Returns the predicted image as a numpy array (flat)
    TODO: maybe just take in patches directly? Will have to change dataloaders...
    """
    cuda = torch.cuda.is_available()
    image_id = image_id.split('_')[0]
    patches = os.listdir(im_patch_dir)
    patch_ids = [f for f in patches if f.split('_')[0] == image_id]

    test_dataset = PatchDataset(patch_ids, im_patch_dir, None, train=False, transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    results, _, _ = normal_predict(unet, test_dataloader, have_labels=True)

    results = results.squeeze()
    results = np.argmax(results, axis=1)

    predicted_im = patching.depatchify(results, results.shape[1:])
    return predicted_im


def batch_predict_full_images(ids_to_predict, patch_dir, model_path, save_pth, 
                              verbose, save_all=True):
    """
    Mostly for testing - assumes we have labels for accuracy score and have already generated patches
    """

    # want to predict on one image at a time so we can reconstruct it
    ids = os.listdir(os.path.join(patch_dir, 'image_patches'))
    image_ids = set([f.split('_')[0] for f in ids]) # set = unique values
    ids_to_predict = [f.split('_')[0] for f in ids_to_predict]
    if verbose:
        print(f"Testing on {len(image_ids)} images with {len(ids)} total patches.")

    # Load the model
    # can change this to accept pre-loaded model if necessary
    unet = utils.load_unet_from_state_dict(model_path)
    im_patch_dir = os.path.join(patch_dir, 'image_patches')
    label_patch_dir = os.path.join(patch_dir, 'label_patches')

    for image_id in tqdm(ids_to_predict, desc='Running U-net on images', disable=not verbose):
        patch_ids = [f for f in ids if f.split('_')[0] == image_id]
        patch_ids = sorted(patch_ids)
        # train = True here because we want to load the labels for accuracy score
        test_dataset = PatchDataset(patch_ids, im_patch_dir, label_patch_dir, train=True, transform=None)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        results, orig, true = normal_predict(unet, test_dataloader, have_labels=True)

        results = results.squeeze()
        results = np.argmax(results, axis=1)

        if save_all:
            save_chance = 1
        else:
            # randomly save some images
            save_chance = np.random.rand()
        if save_chance > 0.9:
            predicted_im = patching.depatchify(results, results.shape[1:])
            # import matplotlib.pyplot as plt
            # plt.imshow(predicted_im)
            # plt.show()
            np.save(os.path.join(save_pth, image_id), predicted_im)


    # TODO calculate DICE

