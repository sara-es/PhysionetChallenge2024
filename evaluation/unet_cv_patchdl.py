import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import pickle, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from digitization import Unet
from digitization.Unet.ECGunet import BasicResUNet
from digitization.Unet.datasets.PatchDataset import PatchDataset
from tqdm import tqdm



def get_data_split_ids(patch_path, patch_size, train_prop, max_samples=False):
    im_patch_path = os.path.join(patch_path, 'image_patches')
    lab_patch_path = os.path.join(patch_path, 'label_patches')
    
    id_train, id_test = Unet.utils.patch_split_from_ids(im_patch_path, lab_patch_path, train_prop, max_samples=max_samples)

    return id_train, id_test, im_patch_path, lab_patch_path



def predict_unet(ids, im_patch_dir, label_patch_dir, args, model_path, save_pth, save_all=True):
    cuda = torch.cuda.is_available()
    print('Testing: ', len(ids), flush=True)
    print('Creating datasets and dataloaders')
    # train = True here because we want to load the labels for accuracy score
    test_dataset = PatchDataset(ids, im_patch_dir, label_patch_dir, train=True, transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load the model
    unet = BasicResUNet(3, 2, nbs=[1, 1, 1, 1], init_channels=16, cbam=False)
    if cuda:
        unet = unet.cuda()
    if model_path:
        print('Loading Weights')
        encoder_dict = unet.state_dict()
        pretrained_dict = torch.load(model_path)['model_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        print('weights loaded unet = ', len(pretrained_dict), '/', len(encoder_dict))
        unet.load_state_dict(torch.load(model_path)['model_state_dict'])
    
    print('Predicting')
    results, true, orig = Unet.normal_predict(args, unet, test_dataloader)

    print(results.shape)
    print(np.unique(results))
    print(np.unique(true))

    results = results.squeeze()
    true = true.squeeze()

    results = np.argmax(results, axis=1)
    true = np.argmax(true, axis=1)

    d_store = []
    for i in range(0, len(results)):
        d = Unet.dice(results[i], true[i])
        print(d)
        if d != 0:
            d_store.append(d)
    d_store = np.array(d_store)
    print('Mean Dice = ', d_store.mean())

    if not save_all:
        np.save(save_pth+'dice_scores', d_store)
        np.save(save_pth+'example_pred', results[0:100])
        np.save(save_pth+'example_true', true[0:100])
        np.save(save_pth+'example_raw', orig[0:100])
    else:
        np.save(save_pth+'dice_scores', d_store)
        np.save(save_pth+'example_pred', results)
        np.save(save_pth+'example_true', true)
        np.save(save_pth+'example_raw', orig)



def main(image_path, labels_path, model_path, output_path, patch_path, train=True, make_patches=True):
    args = Unet.utils.Args()       # This is just a class to pass values efficiently to the training loops
    args.epochs = 50
    args.batch_size = 32
    args.patience = 25  # For the early stopping
    args.train_val_prop = 0.9
    args.learning_rate = 0.5e-3

    reduce_lr = True # Decay the learning rate --> currently just hard coded 

    patchsize = 256 # Assumes square patches
    augmentation = False # True or false 

    # Hard coded these values just to get something running for now
    LOAD_PATH_UNET = None # if we're loading a pretrained model
    PATH_UNET = model_path + 'UNET_' + str(patchsize) # this is where the model will be saved
    CHK_PATH_UNET = model_path + 'UNET_' + str(patchsize) + '_checkpoint' # this is where the model checkpoints will be saved, used with early stopping
    LOSS_PATH = model_path + 'UNET_' + str(patchsize) + '_losses' # this is where the loss values will be saved, used with early stopping

    # reduce the max number of samples because more samples than I have RAM for, 
    # set to False if you have more RAM than me
    # we bypass this by loading patches in dataloaders
    max_samples = False

    if make_patches:
        Unet.patching.save_patches_batch(image_path, labels_path, patchsize, patch_path, max_samples=max_samples)
    print(f'Loading patches from {patch_path}...')

    if train:
        id_train, id_test, im_patch_dir, lab_patch_dir = get_data_split_ids(patch_path, patchsize, 0.8, max_samples=max_samples)
        Unet.train_unet(id_train, im_patch_dir, lab_patch_dir, args,
                PATH_UNET, CHK_PATH_UNET, LOSS_PATH, LOAD_PATH_UNET, 
                max_samples=False, augmentation=augmentation, reduce_lr=reduce_lr
                )
    else:
        im_patch_dir = os.path.join(patch_path, 'image_patches')
        lab_patch_dir = os.path.join(patch_path, 'label_patches')
        id_test = os.listdir(im_patch_dir)
    
    # predict_unet(id_test, im_patch_dir, lab_patch_dir, args,
    #              CHK_PATH_UNET, output_path, 
    #              save_all=True
    #              )
    
    CHK_PATH_UNET = os.path.join('model', 'pretrained', 'UNET_run1_256_checkpoint')
    predict_full_images(id_test, im_patch_dir, lab_patch_dir, args,
                 CHK_PATH_UNET, output_path, 
                 save_all=True
                 )
    

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Visualize the results of the digitization model pipeline on the Challenge data.')
#     parser.add_argument('-i', '--data_folder', type=str, help='The folder containing the Challenge images.')
#     parser.add_argument('-o', '--output_folder', type=str, help='The folder to save the output visualization.')
#     parser.add_argument('-v', '--verbose', action='store_true', help='Print progress messages.')
#     args = parser.parse_args()

#     main(args.data_folder, args.output_folder, args.verbose)

image_path = os.path.join('tiny_testset' , 'lr_unet_tests', 'data_images')
labels_path = os.path.join('tiny_testset' , 'lr_unet_tests', 'binary_masks')
model_path = os.path.join('model')
output_path = os.path.join('tiny_testset' , 'lr_unet_tests')
patch_path = os.path.join('tiny_testset', 'lr_unet_tests', 'unet_outputs') # this is base directory, patches and labels will make their own folders

main(image_path, labels_path, model_path, output_path, patch_path, train=False, make_patches=False)