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


def save_patches(image_path, label_path, patch_size, patch_save_path, max_samples=False):
    ids = sorted(os.listdir(label_path))
    if max_samples:
        ids = ids[:max_samples]
    im_patch_path = os.path.join(patch_save_path, 'image_patches')
    lab_patch_path = os.path.join(patch_save_path, 'label_patches')
    os.makedirs(im_patch_path, exist_ok=True)
    os.makedirs(lab_patch_path, exist_ok=True)

    for id in tqdm(ids):
        lab_pth = label_path + id
        id = id.split('.')[0]
        img_pth = image_path + id + '.png'

        image = plt.imread(img_pth)
        with open(lab_pth, 'rb') as f:
            label = np.load(f)

        im_patches, label_patches = Unet.patchify(image, label, size=(patch_size,patch_size))
        
        for i in range(len(im_patches)):
            im_patch = im_patches[i]
            lab_patch = label_patches[i]
            k = f'_{i:03d}'
            np.save(os.path.join(im_patch_path, id + k), im_patch)
            np.save(os.path.join(lab_patch_path, id + k), lab_patch)
        # np.save(os.path.join(im_patch_path, id), im_patches)
        # np.save(os.path.join(lab_patch_path, id), label_patches)


def patch_split_from_ids(patch_files, train_prop, max_samples=False):
    # there are ~64 patches for each image, shuffle by id to avoid data leakage
    image_ids = set([f.split('_')[0] for f in patch_files]) # set = unique values
    image_ids = shuffle(list(image_ids), random_state=42)

    if max_samples:
        train_prop = int(max_samples * train_prop)
        img_id_train = image_ids[:train_prop]
        img_id_test = image_ids[train_prop:max_samples]
    else:
        n_images = len(image_ids)
        train_prop = int(n_images * train_prop)
        img_id_train = image_ids[:train_prop]
        img_id_test = image_ids[train_prop:]

    # find all patches for each image id in the train and test sets
    id_train = [f for f in patch_files if f.split('_')[0] in img_id_train]
    id_test = [f for f in patch_files if f.split('_')[0] in img_id_test]

    return id_train, id_test


def get_data_split_ids(patch_path, patch_size, train_prop, max_samples=False):
    im_patch_path = os.path.join(patch_path, 'image_patches')
    lab_patch_path = os.path.join(patch_path, 'label_patches')
    patch_files = sorted(os.listdir(im_patch_path))
    
    id_train, id_test = patch_split_from_ids(patch_files, train_prop, max_samples=max_samples)

    return id_train, id_test, im_patch_path, lab_patch_path


def train_unet(ids, im_patch_dir, label_patch_dir, args, PATH_UNET, CHK_PATH_UNET, 
               LOSS_PATH, LOAD_PATH_UNET, max_samples, augmentation, reduce_lr
               ):
    cuda = torch.cuda.is_available()
    train_ids, val_ids = patch_split_from_ids(ids, args.train_val_prop, max_samples=max_samples)

    print('Training: ', len(train_ids), flush=True)
    print('Validation: ', len(val_ids), flush=True)

    print('Creating datasets and dataloaders')
    train_dataset = PatchDataset(train_ids, im_patch_dir, label_patch_dir, transform=augmentation)
    val_dataset = PatchDataset(val_ids, im_patch_dir, label_patch_dir, transform=None)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load the model
    unet = BasicResUNet(3, 2, nbs=[1, 1, 1, 1], init_channels=16, cbam=False)
    if cuda:
        unet = unet.cuda()
    if LOAD_PATH_UNET:
        print('Loading Weights')
        encoder_dict = unet.state_dict()
        pretrained_dict = torch.load(LOAD_PATH_UNET)['model_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        print('weights loaded unet = ', len(pretrained_dict), '/', len(encoder_dict))
        unet.load_state_dict(torch.load(LOAD_PATH_UNET)['model_state_dict'])
    
    optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
    early_stopping = Unet.utils.EarlyStopping(args.patience, verbose=False)

    # Loss function from paper --> dice loss + focal loss
    criterion = Unet.ComboLoss(
                weights={'dice': 1, 'focal': 1},
                channel_weights=[1],
                channel_losses=[['dice', 'focal']],
                per_image=False
            )
    if cuda:
        crit = criterion.cuda()
        
    epoch_reached = 1
    loss_store = []

    for epoch in range(epoch_reached, args.epochs+1):
        print('Epoch ', epoch, '/', args.epochs, flush=True)
        loss = Unet.train_normal(args, unet, train_dataloader, optimizer, crit, epoch)
        val_loss = Unet.val_normal(args, unet, val_dataloader, crit)
        loss_store.append([loss, val_loss])
        np.save(LOSS_PATH, np.array(loss_store))

        # Decide whether the model should stop training or not
        early_stopping(val_loss, unet, epoch, optimizer, loss, CHK_PATH_UNET)

        if early_stopping.early_stop:
            loss_store = np.array(loss_store)
            np.save(LOSS_PATH, loss_store)
            break
            
        if reduce_lr:
            if early_stopping.counter == 5:
                print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
            if early_stopping.counter == 10:
                print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
            if early_stopping.counter == 15:
                print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
            if early_stopping.counter == 20:
                print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
                        
        if epoch == args.epochs:
            print('Finished Training', flush=True)
            print('Saving the model', flush=True)

            # Save the model in such a way that we can continue training later
            torch.save(unet.state_dict(), PATH_UNET)
            loss_store = np.array(loss_store)
            np.save(LOSS_PATH, loss_store)

        torch.cuda.empty_cache()  # Clear memory cache


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


def predict_full_images(ids, im_patch_dir, label_patch_dir, args, model_path, save_pth, save_all=True):
    cuda = torch.cuda.is_available()

    # want to predict on one image at a time so we can reconstruct it
    image_ids = set([f.split('_')[0] for f in ids]) # set = unique values
    print(f"Testing on {len(image_ids)} images with {len(ids)} total patches.")

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

    for image_id in tqdm(image_ids):
        patch_ids = [f for f in ids if f.split('_')[0] == image_id]
        patch_ids = sorted(patch_ids)
        # train = True here because we want to load the labels for accuracy score
        test_dataset = PatchDataset(patch_ids, im_patch_dir, label_patch_dir, train=True, transform=None)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        results, true, orig = Unet.normal_predict(args, unet, test_dataloader)
        results = results.squeeze()
        results = np.argmax(results, axis=1)

        if save_all:
            save_chance = 1
        else:
            # randomly save some images
            save_chance = np.random.rand()
        if save_chance > 0.9:
            predicted_im = Unet.depatchify(results, results.shape[1:])
            # import matplotlib.pyplot as plt
            # plt.imshow(predicted_im)
            # plt.show()
            np.save(save_pth + image_id, predicted_im)

    # print(results.shape)
    # print(np.unique(results))
    # print(np.unique(true))

    results = results.squeeze()
    true = true.squeeze()

    results = np.argmax(results, axis=1)
    true = np.argmax(true, axis=1)

    d_store = []
    for i in range(0, len(results)):
        d = Unet.dice(results[i], true[i])
        # print(d)
        if d != 0:
            d_store.append(d)
    d_store = np.array(d_store)
    print('Mean Dice = ', d_store.mean())


def main(image_path, labels_path, model_path, output_path):
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
    PATH_UNET = model_path + '_' + str(patchsize) # this is where the model will be saved
    CHK_PATH_UNET = model_path + '_' + str(patchsize) + '_checkpoint' # this is where the model checkpoints will be saved, used with early stopping
    LOSS_PATH = model_path + '_' + str(patchsize) + '_losses' # this is where the loss values will be saved, used with early stopping
    PATCH_PATH = 'F:\\ptb-xl-indiv'

    # reduce the max number of samples because more samples than I have RAM for, 
    # set to False if you have more RAM than me
    # we bypass this by loading patches in dataloaders
    max_samples = 50 # False

    # save_patches(image_path, labels_path, patchsize, PATCH_PATH, max_samples=max_samples)
    print(f'Loading patches from {PATCH_PATH}...')
    id_train, id_test, im_patch_dir, lab_patch_dir = get_data_split_ids(PATCH_PATH, patchsize, 0.8, max_samples=max_samples)

    # train_unet(id_train, im_patch_dir, lab_patch_dir, args,
    #            PATH_UNET, CHK_PATH_UNET, LOSS_PATH, LOAD_PATH_UNET, 
    #            max_samples=False, augmentation=augmentation, reduce_lr=reduce_lr
    #            )
    
    # predict_unet(id_test, im_patch_dir, lab_patch_dir, args,
    #              CHK_PATH_UNET, output_path, 
    #              save_all=True
    #              )
    
    CHK_PATH_UNET = 'G:\\PhysionetChallenge2024\\model\\pretrained\\UNET_run1_256_checkpoint'
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

image_path = 'G:\\PhysionetChallenge2024\\ptb-xl\\combined_records_img1\\'
labels_path = 'G:\\PhysionetChallenge2024\\ptb-xl\\binary_masks\\'
model_path = 'G:\\PhysionetChallenge2024\\model\\UNET' # no \ at the end of this one
output_path = 'G:\\PhysionetChallenge2024\\ptb-xl\\img1_output\\'

main(image_path, labels_path, model_path, output_path)