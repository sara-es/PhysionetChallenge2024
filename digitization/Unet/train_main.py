# Adapted from code by Nicola Dinsdale 2024
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from digitization.Unet.ECGunet import BasicResUNet
from digitization.Unet.datasets.PatchDataset import PatchDataset
from digitization.Unet import utils
from digitization.Unet import Unet


def train_epoch(args, model, train_loader, optimizer, criterion, epoch):
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

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx+1) / len(train_loader), loss.item()), flush=True)
            del loss
    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())

    del av_loss
    print('\nTraining set: Average loss: {:.4f}'.format(av_loss_copy,  flush=True))
    return av_loss_copy


def val_epoch(args, model, val_loader, criterion):
    cuda = torch.cuda.is_available()
    model.eval()
    total_loss = 0
    batches = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target.type(torch.LongTensor)
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            batches += 1
            x = model(data)
            loss = criterion(x, target)
            total_loss  += loss
            # example = x[0:5].detach().cpu().numpy()
            # np.save('G:\\PhysionetChallenge2024\\evaluation\\viz\\unet\\example_', example)
    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())
    del av_loss
    return av_loss_copy


def train_unet(ids, im_patch_dir, label_patch_dir, args, 
               PATH_UNET, CHK_PATH_UNET, LOSS_PATH, LOAD_PATH_UNET, verbose, 
               max_samples=False,
               ):
    cuda = torch.cuda.is_available()

    train_patch_ids, val_patch_ids = utils.patch_split_from_ids(ids, im_patch_dir, label_patch_dir, 
                                                    args.train_val_prop, max_samples=max_samples)

    if verbose:
        print('Training patches: ', len(train_patch_ids), flush=True)
        print('Validation patches: ', len(val_patch_ids), flush=True)

        print('Creating datasets and dataloaders')
    train_dataset = PatchDataset(train_patch_ids, im_patch_dir, label_patch_dir, transform=args.augmentation)
    val_dataset = PatchDataset(val_patch_ids, im_patch_dir, label_patch_dir, transform=None)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load the model
    unet = BasicResUNet(3, 2, nbs=[1, 1, 1, 1], init_channels=16, cbam=False)
    if cuda:
        unet = unet.cuda()
    if LOAD_PATH_UNET:
        if verbose:
            print('Loading Weights')
        encoder_dict = unet.state_dict()
        pretrained_dict = torch.load(LOAD_PATH_UNET)['model_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        if verbose:  
            print('weights loaded unet = ', len(pretrained_dict), '/', len(encoder_dict))
        unet.load_state_dict(torch.load(LOAD_PATH_UNET)['model_state_dict'])
    
    optimizer = optim.AdamW(lr=args.learning_rate, params=unet.parameters())
    early_stopping = utils.EarlyStopping(args.patience, verbose=False)

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
        if verbose:
            print('Epoch ', epoch, '/', args.epochs, flush=True)
        loss = Unet.train_epoch(args, unet, train_dataloader, optimizer, crit, epoch)
        if args.train_val_prop < 1.0:
            val_loss = Unet.val_epoch(args, unet, val_dataloader, crit)
        else:
            val_loss = 0
        if verbose:
            print('Validation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
        loss_store.append([loss, val_loss])
        np.save(LOSS_PATH, np.array(loss_store))

        # Decide whether the model should stop training or not
        early_stopping(val_loss, unet, epoch, optimizer, loss, CHK_PATH_UNET)

        if early_stopping.early_stop:
            loss_store = np.array(loss_store)
            np.save(LOSS_PATH, loss_store)
            break
            
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
                print('Finished Training', flush=True)
                print('Saving U-net model', flush=True)

            # Save the model in such a way that we can continue training later
            torch.save(unet.state_dict(), PATH_UNET)
            loss_store = np.array(loss_store)
            np.save(LOSS_PATH, loss_store)

        torch.cuda.empty_cache()  # Clear memory cache