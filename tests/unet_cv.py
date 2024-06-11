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
from digitization.Unet.datasets import numpy_dataset
from tqdm import tqdm


def save_patches(image_path, label_path, patch_size, patch_save_path, max_samples=None):
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
        
        np.save(os.path.join(im_patch_path, id), im_patches)
        np.save(os.path.join(lab_patch_path, id), label_patches)


def load_patches_data_split(patch_path, patch_size, train_prop, load_all=True):
    # TODO: don't need patch size here, can infer when loading
    im_patch_path = os.path.join(patch_path, 'image_patches')
    lab_patch_path = os.path.join(patch_path, 'label_patches')
    ids = sorted(os.listdir(im_patch_path))
    ids = shuffle(ids)
    n_images = len(ids)

    if not load_all:
        # cap at 100 images
        train_prop = int(100 * train_prop)
        id_train = ids[:train_prop]
        id_test = ids[train_prop:100]
    else:
        train_prop = int(n_images * train_prop)
        id_train = ids[:train_prop]
        id_test = ids[train_prop:]

    # quite slow concating these, but much more memory efficient than lists
    # if we knew how many patches per image in advance this would be a lot faster
    X_train = np.zeros((1, patch_size, patch_size, 3)) 
    y_train = np.zeros((1, patch_size, patch_size))
    print(f"Loading {len(id_train)} training images...")
    for id in tqdm(id_train):
        image = np.load(os.path.join(im_patch_path, id))
        label = np.load(os.path.join(lab_patch_path, id))

        X_train = np.append(X_train, image, axis=0)
        y_train = np.append(y_train, label, axis=0)

    X_test = np.zeros((1, patch_size, patch_size, 3))
    y_test = np.zeros((1, patch_size, patch_size))
    print(f"Loading {len(id_test)} testing images...")
    for id in tqdm(id_test):
        image = np.load(os.path.join(im_patch_path, id))
        label = np.load(os.path.join(lab_patch_path, id))

        X_test = np.append(X_test, image)
        y_test = np.append(y_test, label)

    X_test = X_test.reshape(-1, patch_size, patch_size, 3)
    y_test = y_test.reshape(-1, patch_size, patch_size)
    X_train = X_train.reshape(-1, patch_size, patch_size, 3)
    y_train = y_train.reshape(-1, patch_size, patch_size)

    return X_train, y_train, X_test, y_test


def prepare_data_split(image_path, label_path, patch_size, train_prop):
    ids = sorted(os.listdir(label_path))
    print(ids)
    ids = shuffle(ids)

    # train_prop = int(len(id) * train_prop)
    # id_train = id[:train_prop]
    # id_test = id[train_prop:]

    # test small proportion of data
    train_prop = int(len(ids) * 0.002)
    test_prop = int(len(ids) * 0.0022)
    id_train = ids[:train_prop]
    id_test = ids[train_prop:test_prop]

    print('Train Subjects = ', len(id_train))
    print('Test Subjects = ', len(id_test))

    X_train = np.zeros((1, patch_size, patch_size, 3))
    y_train = np.zeros((1, patch_size, patch_size))
    for id in tqdm(id_train):
        # print(id, flush=True)
        lab_pth = label_path + id
        id = id.split('.')[0]
        img_pth = image_path + id + '.png'

        image = plt.imread(img_pth)
        with open(lab_pth, 'rb') as f:
            label = pickle.load(f)
            np.save(lab_pth, label)

        im_patches, label_patches = Unet.patchify(image, label, size=(patch_size,patch_size))
        X_train = np.append(X_train, im_patches)
        y_train = np.append(y_train, label_patches)

    X_train = np.array(X_train).reshape(-1, patch_size, patch_size, 3 )
    # X_train = np.transpose(X_train, (0,3,1,2))
    y_train = np.array(y_train).reshape(-1, patch_size, patch_size)

    X_test = np.zeros((1, patch_size, patch_size, 3))
    y_test = np.zeros((1, patch_size, patch_size))
    for id in tqdm(id_test):
        lab_pth = label_path + id
        id = id.split('.')[0]
        img_pth = image_path + id + '.png'

        image = plt.imread(img_pth)
        with open(lab_pth, 'rb') as f:
            label = pickle.load(f)

        im_patches, label_patches = Unet.patchify(image, label, size=(patch_size,patch_size))
        X_test = np.append(X_test, im_patches)
        y_test = np.append(y_test, label_patches)

    X_test = np.array(X_test).reshape(-1, patch_size, patch_size, 3 )
    # X_test = np.transpose(X_test, (0,3,1,2))
    y_test = np.array(y_test).reshape(-1, patch_size, patch_size)

    return X_train, y_train, X_test, y_test


def train_unet(X_train, y_train, args, patchsize, PATH_UNET, CHK_PATH_UNET, 
               LOSS_PATH, LOAD_PATH_UNET, max_samples, augmentation, reduce_lr
               ):
    cuda = torch.cuda.is_available()
    X = np.transpose(X_train, (0,3,1,2))
    y = y_train.reshape(-1, patchsize, patchsize)

    X, y = shuffle(X, y, random_state=0)

    proportion = int(args.train_val_prop * len(X))
    X_train, y_train = X[:proportion], y[:proportion]
    X_val, y_val = X[proportion:], y[proportion:]

    if max_samples:
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]

    print('Training: ', X_train.shape, y_train.shape, flush=True)
    print('Validation: ', X_val.shape, y_val.shape, flush=True)

    # Make the training labels one hot
    y_store = np.zeros((2, y_train.shape[0], patchsize, patchsize))
    print(y_store.shape)
    print(np.unique(y_train))
    print(np.unique(y_val))
    y_store[0][y_train==0] = 1
    y_store[1][y_train==1] = 1
    y_train = y_store
    y_train = np.transpose(y_train, (1, 0, 2, 3))
    print(y_train.shape)

    y_store = np.zeros((2, y_val.shape[0], patchsize, patchsize))
    y_store[0][y_val==0] = 1
    y_store[1][y_val==1] = 1
    y_val = y_store
    y_val = np.transpose(y_val, (1, 0, 2, 3))

    print('Training: ', X_train.shape, y_train.shape, flush=True)
    print('Validation: ', X_val.shape, y_val.shape, flush=True)

    print('Creating datasets and dataloaders')
    train_dataset = numpy_dataset.numpy_dataset(X_train, y_train, transform=augmentation)
    val_dataset = numpy_dataset.numpy_dataset(X_val, y_val, transform=augmentation)

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
        loss = Unet.train_epoch(args, unet, train_dataloader, optimizer, crit, epoch)
        val_loss = Unet.val_epoch(args, unet, val_dataloader, crit)
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


def predict_unet(X_test, y_test, args, patch_size, model_path, save_pth, save_all=True):
    cuda = torch.cuda.is_available()
    X = np.transpose(X_test, (0,3,1,2))
    y = y_test.reshape(-1, patch_size, patch_size)

    # Make the training labels one hot
    y_store = np.zeros((2, y.shape[0], patch_size, patch_size))
    print(np.unique(y))
    print(np.unique(y))
    y_store[0][y==0] = 1
    y_store[1][y==1] = 1
    y = y_store
    y = np.transpose(y, (1, 0, 2, 3))
    print(y.shape)

    print(np.unique(y))


    print('Testing: ', X.shape, y.shape, flush=True)

    print('Creating datasets and dataloaders')
    test_dataset = numpy_dataset.numpy_dataset(X, y)
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



def main(image_path, labels_path, model_path, output_path):
    args = Unet.utils.Args()       # This is just a class to pass values efficiently to the training loops
    args.epochs = 10
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
    PATCH_PATH = 'F:\\ptb-xl'

    # reduce the max number of samples because more samples than I have RAM for, 
    # set to False if you have more RAM than me
    # could bypass this by loading patches in dataloaders
    max_samples = 100 

    # X_train, y_train, X_test, y_test = prepare_data_split(image_path, labels_path, patchsize, 0.8)

    # save_patches(image_path, labels_path, patchsize, PATCH_PATH, max_samples=max_samples)
    print(f'Loading patches from {PATCH_PATH}...')
    X_train, y_train, X_test, y_test = load_patches_data_split(PATCH_PATH, patchsize, 0.8, load_all=False)

    # np.save(save_pth + 'X_train_' + str(patch_size), X_train)
    # np.save(save_pth + 'y_train_' + str(patch_size), y_train)
    # np.save(save_pth + 'X_test_' + str(patch_size), X_test)
    # np.save(save_pth + 'y_test_' + str(patch_size), y_test)

    train_unet(X_train, y_train, args, patchsize, 
               PATH_UNET, CHK_PATH_UNET, LOSS_PATH, LOAD_PATH_UNET, 
               max_samples, augmentation, reduce_lr
               )
    
    predict_unet(X_test, y_test, args, patchsize,
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