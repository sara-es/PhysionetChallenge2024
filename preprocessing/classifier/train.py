import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessing.classifier import datasets, utils
from preprocessing.classifier.ResNet_adapt import ResNet_adapt
from utils import constants


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

    data = data.detach().cpu().numpy()
    pred = x.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    del av_loss
    if verbose:
        print('\nTraining set: Average loss: {:.4f}'.format(av_loss_copy,  flush=True))
    return av_loss_copy, data, pred, target


def val_epoch(args, model, val_loader, criterion, verbose):
    cuda = torch.cuda.is_available()
    model.eval()
    true_store = []
    pred_store = []
    loss_store = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target.type(torch.LongTensor)
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            x = model(data)
            loss = criterion(x, target)
            true_store.append(target.detach().cpu().numpy())
            pred_store.append(x.detach().cpu().numpy())
            loss_store.append(loss.detach().cpu().numpy())

    true_store = np.array(true_store).squeeze()
    pred_store = np.array(pred_store).squeeze()
    pred_store = np.argmax(pred_store, axis=1)
    accuracy = accuracy_score(true_store, pred_store)
    total_loss = np.mean(loss_store)
    if verbose:
        print('Validation set: Accuracy: {:.4f}\n'.format(accuracy,  flush=True))
    return accuracy, total_loss


def train_image_classifier(real_patch_folder, gen_patch_folder, model_folder, 
                                      patch_size, verbose, args=None):
    if not args: 
        args = utils.Args()
        args.learning_rate = 0.5e-3
        args.patience = 10
        args.epochs = constants.UNET_EPOCHS

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    r_patch_folder = os.path.join(real_patch_folder, 'image_patches')
    g_patch_folder = os.path.join(gen_patch_folder, 'image_patches')
    train_data, train_labels, val_data, val_labels = utils.prepare_classifier_data(
                                        r_patch_folder, g_patch_folder, 
                                        args.train_val_prop, verbose, 
                                        delete_images=False)

    if verbose:
        print('Training patches: ', len(train_data), flush=True)
        print('Validation patches: ', len(val_data), flush=True)

        print('Creating datasets and dataloaders...')
    workers = 8 if cuda and constants.ALLOW_MULTIPROCESSING else 0
    train_dataset = datasets.PatchDataset(train_data, train_labels, #transform=None)
                                 transform=args.augmentation)
    val_dataset = datasets.PatchDataset(val_data, val_labels, transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=workers, 
                                  pin_memory=(True if device == 'cuda' else False))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers,
                                pin_memory=(True if device == 'cuda' else False))

    # Initialize and load the model
    # TODO will have to download this in advance - won't run in Challenge environment
    model = ResNet_adapt(embsize=2, weights='DEFAULT')
    epoch_reached = 1
    loss_store = []  
    optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())
    early_stopping = utils.EarlyStopping(args.patience, verbose=verbose)

    if cuda:
        model = model.cuda()
    model.to(device)
        
    # if LOAD_PATH_UNET:
    #     if verbose:
    #         print('Loading U-net checkpoint...', flush=True)
    #     try:
    #         checkpoint = torch.load(LOAD_PATH_UNET)
    #         unet.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         epoch_reached = checkpoint['epoch']
    #         if verbose:
    #             print(f'{len(checkpoint["model_state_dict"])}/{len(unet.state_dict())} weights ' +\
    #                     f'loaded from {LOAD_PATH_UNET}. Starting from epoch {epoch_reached}.',
    #                     flush=True)
    #     except Exception as e:
    #         print(e)
    #         print(f'Could not load U-net checkpoint from {LOAD_PATH_UNET}. '+\
    #                'Training from scratch...', flush=True)
    #     try:
    #         with open(LOSS_PATH + '.npy', 'rb') as f:
    #             loss_store = np.load(f, allow_pickle=True)
    #         loss_store = loss_store.tolist()
    #         val_losses = np.array([x[1] for x in loss_store if x[1] is not None])
    #         early_stopping.best_score = -np.min(val_losses)
    #         early_stopping.val_loss_min = np.min(val_losses)
    #     except Exception as e:
    #         print(e)
    #         print(f'Could not load loss history from {LOSS_PATH}. Early stopping will be delayed.',
    #               flush=True)

    if epoch_reached > args.epochs: 
        # checkpoint already reached the desired number of epochs, make sure we still return the model        
        return model.state_dict()
            
    criterion = nn.CrossEntropyLoss()
    if cuda:
        crit = criterion.cuda()
    else:
        crit = criterion
    crit.to(device)

    for epoch in range(epoch_reached, args.epochs+1):
        if verbose:
            print('Epoch ', epoch, '/', args.epochs, flush=True)
        loss, data, pred, true = train_epoch(args, model, train_dataloader, optimizer, crit, epoch, verbose)
        if args.train_val_prop < 1.0:
            val_accuracy, val_loss = val_epoch(args, model, val_dataloader, criterion, verbose)
            if verbose:
                print('Validation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
        else:
            val_loss = 0
        loss_store.append([loss, val_loss])
        # np.save(LOSS_PATH, np.array(loss_store))

        # # Save some example images, for debugging
        # pred_class = pred
        # true_class = true
        # orig_patches = data.transpose(0, 2, 3, 1).squeeze()

        # import matplotlib.pyplot as plt

        # for patch in range(len(pred)):
        #     fig, ax = plt.subplots(1, 1, figsize=(5, 5), num=patch, clear=True)
        #     ax.imshow(orig_patches[patch])
        #     title = ''
        #     if true_class[patch] == 0:
        #         title += 'Generated image, '
        #     else:
        #         title += 'Real image, '
        #     title += 'Predicted: '
        #     if pred_class[patch] == 0:
        #         title += 'Generated'
        #     else:
        #         title += 'Real'
        #     ax.set_title(title)
        #     # save the plot
        #     results_path = os.path.join("G:\\PhysionetChallenge2024", "temp_data", "cls_patch_results")
        #     plt.savefig(os.path.join(results_path, 'epoch_' + str(epoch) +  '-' + str(patch) + '.png'))
        #     plt.close()

        # Decide whether the model should stop training or not
        CHK_PATH_CLSFR = os.path.join(model_folder, 'classifier_' + str(patch_size))
        early_stopping(val_loss, model, epoch, optimizer, loss, CHK_PATH_CLSFR)

        if early_stopping.early_stop:
            loss_store = np.array(loss_store)
            # np.save(LOSS_PATH, loss_store)
            torch.save(model.state_dict(), CHK_PATH_CLSFR)
            return model.state_dict()
            
        if args.reduce_lr:
            if early_stopping.counter == 2:
                if verbose:
                    print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())
            if early_stopping.counter == 4:
                if verbose:
                    print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())
            if early_stopping.counter == 6:
                if verbose:
                    print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())
            if early_stopping.counter == 8:
                if verbose:
                    print('Reducing learning rate')
                args.learning_rate = args.learning_rate/2
                optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())
                        
        if epoch == args.epochs:
            if verbose:
                print('Finished Training. Saving U-net model...', flush=True)

            # Save the model in such a way that we can continue training later
            loss_store = np.array(loss_store)
            # np.save(LOSS_PATH, loss_store)
            torch.save(model.state_dict(), CHK_PATH_CLSFR)
            return model.state_dict()
            
        torch.cuda.empty_cache()  # Clear memory cache