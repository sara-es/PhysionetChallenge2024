import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score 
import torch.optim as optim
from torch.utils.data import DataLoader

from digitization.Unet.ECGunet import BasicResUNet
from preprocessing.classifier import PatchDataset
from digitization.Unet import utils
from digitization.Unet import Unet


def train_epoch(args, model, train_loader, optimizer, criterion, epoch, verbose):
    cuda = torch.cuda.is_available()
    total_loss = 0
    model.train()
    batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        if list(data.size())[0] == args.batch_size :
            batches += 1

            # First update the encoder and regressor
            optimizer.zero_grad()
            x = model(data) 
            loss = criterion(x, target)
            loss.backward()
            optimizer.step()

            total_loss += loss

            if verbose and batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx+1) / len(train_loader), loss.item()), flush=True)
            del loss
    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())

    del av_loss
    if verbose:
        print('\nTraining set: Average loss: {:.4f}'.format(av_loss_copy,  flush=True))
    return av_loss_copy


def val_epoch(args, model, val_loader, verbose):
    cuda = torch.cuda.is_available()
    model.eval()
    true_store = []
    pred_store = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            x = model(data)
            true_store.append(target.detach().cpu().numpy())
            pred_store.append(x.detach().cpu().numpy())

    true_store = np.array(true_store).squeeze()
    pred_store = np.array(pred_store).squeeze()
    pred_store = np.argmax(pred_store, axis=1)
    accuracy = accuracy_score(true_store, pred_store)
    if verbose:
        print('Validation set: Accuracy: {:.4f}\n'.format(accuracy,  flush=True))
    return accuracy


def train_image_classifier(ids, generated_patch_dir, real_patch_dir, args, 
               CHK_PATH_UNET, LOSS_PATH, LOAD_PATH_UNET, verbose, 
               max_samples=False,):
    """
    Note max_samples is number of PATCHES to use, not images.
    """
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    train_patch_ids, val_patch_ids = utils.patch_split_from_ids(ids, im_patch_dir, label_patch_dir,
                                        args.train_val_prop, verbose, max_samples=max_samples)

    if verbose:
        print('Training patches: ', len(train_patch_ids), flush=True)
        print('Validation patches: ', len(val_patch_ids), flush=True)

        print('Creating datasets and dataloaders...')
    train_dataset = PatchDataset(train_patch_ids, im_patch_dir, label_patch_dir, 
                                 transform=args.augmentation)
    val_dataset = PatchDataset(val_patch_ids, im_patch_dir, label_patch_dir, transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=8, pin_memory=(True if device == 'cuda' else False))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, 
                                pin_memory=(True if device == 'cuda' else False))

    # Initialize and load the model
    unet = BasicResUNet(3, 2, nbs=[1, 1, 1, 1], init_channels=16, cbam=False)
    epoch_reached = 1
    loss_store = []  
    optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
    early_stopping = utils.EarlyStopping(args.patience, verbose=verbose)

    if cuda:
        unet = unet.cuda()
    unet.to(device)
        
    if LOAD_PATH_UNET:
        if verbose:
            print('Loading U-net checkpoint...', flush=True)
        try:
            checkpoint = torch.load(LOAD_PATH_UNET)
            unet.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_reached = checkpoint['epoch']
            if verbose:
                print(f'{len(checkpoint["model_state_dict"])}/{len(unet.state_dict())} weights ' +\
                        f'loaded from {LOAD_PATH_UNET}. Starting from epoch {epoch_reached}.',
                        flush=True)
        except Exception as e:
            print(e)
            print(f'Could not load U-net checkpoint from {LOAD_PATH_UNET}. '+\
                   'Training from scratch...', flush=True)
        try:
            with open(LOSS_PATH + '.npy', 'rb') as f:
                loss_store = np.load(f, allow_pickle=True)
            loss_store = loss_store.tolist()
            val_losses = np.array([x[1] for x in loss_store if x[1] is not None])
            early_stopping.best_score = -np.min(val_losses)
            early_stopping.val_loss_min = np.min(val_losses)
        except Exception as e:
            print(e)
            print(f'Could not load loss history from {LOSS_PATH}. Early stopping will be delayed.',
                  flush=True)

    if epoch_reached > args.epochs: 
        # checkpoint already reached the desired number of epochs, make sure we still return the model        
        return unet.state_dict()
            
    # Loss function from paper --> dice loss + focal loss
    criterion = Unet.ComboLoss(
                weights={'dice': 1, 'focal': 1},
                channel_weights=[1],
                channel_losses=[['dice', 'focal']],
                per_image=False
            )
    if cuda:
        crit = criterion.cuda()
    crit.to(device)

    for epoch in range(epoch_reached, args.epochs+1):
        if verbose:
            print('Epoch ', epoch, '/', args.epochs, flush=True)
        loss, pred, orig, true = Unet.train_epoch(args, unet, train_dataloader, optimizer, crit, epoch, verbose)
        if args.train_val_prop < 1.0:
            val_loss = Unet.val_epoch(args, unet, val_dataloader, crit, epoch, verbose)
            if verbose:
                print('Validation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
        else:
            val_loss = 0
        loss_store.append([loss, val_loss])
        np.save(LOSS_PATH, np.array(loss_store))

        # # Save some example images, for debugging
        # pred_patches = np.argmax(pred.squeeze(), axis=1)
        # true_patches = np.argmax(true.squeeze(), axis=1)
        # orig_patches = orig.squeeze().transpose(0, 2, 3, 1)

        # for patch in range(orig.shape[0]):
        #     fig, ax = plt.subplots(1, 3, figsize=(15, 5), num=patch, clear=True)
        #     ax[0].imshow(pred_patches[patch], cmap='gray')
        #     ax[0].set_title('Predicted Image')
        #     ax[1].imshow(true_patches[patch], cmap='gray')
        #     ax[1].set_title('True Image')
        #     ax[2].imshow(orig_patches[patch])
        #     ax[2].set_title('Original Image')
        #     # save the plot
        #     results_path = os.path.join("G:\\PhysionetChallenge2024", "temp_data", "patch_results")
        #     plt.savefig(os.path.join(results_path, 'epoch_' + str(epoch) +  '-' + str(patch) + '.png'))
        #     plt.close()

        # Decide whether the model should stop training or not
        early_stopping(val_loss, unet, epoch, optimizer, loss, CHK_PATH_UNET)

        if early_stopping.early_stop:
            loss_store = np.array(loss_store)
            np.save(LOSS_PATH, loss_store)
            torch.save(unet.state_dict(), CHK_PATH_UNET)
            return unet.state_dict()
            
        if args.reduce_lr:
            if early_stopping.counter == 5:
                if verbose:
                    print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
            if early_stopping.counter == 10:
                if verbose:
                    print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
            if early_stopping.counter == 15:
                if verbose:
                    print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
            if early_stopping.counter == 20:
                if verbose:
                    print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
                        
        if epoch == args.epochs:
            if verbose:
                print('Finished Training. Saving U-net model...', flush=True)

            # Save the model in such a way that we can continue training later
            loss_store = np.array(loss_store)
            np.save(LOSS_PATH, loss_store)
            torch.save(unet.state_dict(), CHK_PATH_UNET)
            return unet.state_dict()
            
        torch.cuda.empty_cache()  # Clear memory cache