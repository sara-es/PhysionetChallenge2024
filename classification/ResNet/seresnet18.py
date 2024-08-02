import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset

from classification.ResNet.SEResNet import ResNet, BasicBlock
from classification.ResNet.datasets.ECGDataset import ECGDataset, CustomECGDataset
from classification import utils
from utils.model_persistence import save_model_torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold 

def train(model, train_loader, device, loss_fct, sigmoid, optimizer, epoch, uniq_labels, verbose):
    """
    
    """
    model.train()

    if verbose: # only need to keep track of this if we're printing it out
        running_loss = 0.0
        batches_per_printout = 20
        labels_all = torch.tensor((), device=device)
        logits_prob_all = torch.tensor((), device=device)

    with torch.set_grad_enabled(True):    
        for batch_idx, (ecgs, ag, labels) in enumerate(train_loader):
            ecgs = ecgs.float().to(device) # ECGs
            ag = ag.float().to(device) # demographics
            labels = labels.float().to(device) # diagnoses 

            optimizer.zero_grad()
            logits = model(ecgs, ag) 
            loss = loss_fct(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Optional: print training information
            if verbose:
                running_loss += loss.item() # CHECK THIS: this was previously loss.item() * ecgs.size(0)
                logits_prob_all = torch.cat((logits_prob_all, sigmoid(logits)), 0)  
                labels_all = torch.cat((labels_all, labels), 0)

                # Accumulate loss and accuracy for n batches, print, then zero running loss
                if batch_idx % batches_per_printout == batches_per_printout-1:
                    # Approximate calculation of average loss per sample
                    avg_loss = running_loss / (batches_per_printout * len(ecgs))
                    f_measure = utils.compute_classification_metrics(
                                    labels_all, logits_prob_all, uniq_labels, threshold=0.5)
                    
                    # Print
                    print(f'Epoch {epoch} [{(batch_idx+1) * len(ecgs)}/{len(train_loader.dataset)}] \
                          loss: {avg_loss:.4f}, F-measure: {f_measure:.4f}')
                    
                    # Reset accumulated loss and outputs
                    running_loss = 0.0
                    labels_all = torch.tensor((), device=device)
                    logits_prob_all = torch.tensor((), device=device)


def eval(model, train_loader, device, loss_fct, sigmoid, epoch, uniq_labels, verbose):
    model.eval()
    epoch_loss = 0.0
    labels_all = torch.tensor((), device=device)
    logits_prob_all = torch.tensor((), device=device)

    with torch.no_grad():
        for i, (ecgs, ag, labels) in enumerate(train_loader):
            ecgs = ecgs.float().to(device)
            ag = ag.float().to(device) # demographics
            labels = labels.float().to(device) # diagnoses 

            # Run model
            logits = model(ecgs, ag) 
            logits_prob = sigmoid(logits)

            # Calculate loss, append outputs to tensors
            epoch_loss += loss_fct(logits, labels)
            logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)  
            labels_all = torch.cat((labels_all, labels), 0)

    f_measure = utils.compute_classification_metrics(
                        labels_all, logits_prob_all, uniq_labels, threshold=0.5)
    if verbose:
        epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}, val loss: {epoch_loss:.4f}, F-measure: {f_measure:.4f}')
    
    return f_measure


def test(model, test_loader, device, sigmoid, verbose):
    model.eval()
    logits_prob_all = torch.tensor((), device=device)  

    with torch.no_grad():
        for i, (ecgs, ag) in enumerate(test_loader):
            ecgs = ecgs.float().to(device) # ECGs
            ag = ag.float().to(device) # age and gender

            logits = model(ecgs, ag)
            logits_prob = sigmoid(logits)
            logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)
        
    torch.cuda.empty_cache()

    return logits_prob_all.cpu().detach().numpy().squeeze()


def train_model(data, multilabels, uniq_labels, args, verbose):
    """
    Parameters:
        data (list): list of data where [path (str), fs (int), age and sex features (np.array)]
        multilabels (list): List of multilabels, one hot encoded
        uniq_labels (list): List of unique labels as strings
        args (Resnet_args object): Params for the resnet model
        verbose (bool): printouts?
    Returns:
        state_dict (pytorch model): state dictionary of the trained model
        metrics (float): F-measure (if validate=True, else None)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print('Cuda available, using GPU(s).')
    else:
        device = torch.device("cpu")
        if verbose:
            print('Cuda not found. Using CPU.')

    model = ResNet(BasicBlock, [2, 2, 2, 2], 
                in_channel=args.in_channels, 
                out_channel=len(uniq_labels))

    optimizer = optim.Adam(model.parameters(), 
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)
    model.to(device)

    custom_dataset = CustomECGDataset(data, multilabels)

    if args.kfold:
        if verbose:
            print(f'Performing {args.n_splits}-fold cross-validation...')

        idx = np.arange(len(data))
        skfold = MultilabelStratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=2024)
        
        metrics = []
        for i, (train_idx, val_idx) in enumerate(skfold.split(idx, multilabels)):
    
            train_subset = Subset(custom_dataset, train_idx)
            train_dl = DataLoader(train_subset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=1,
                                pin_memory=(True if device == 'cuda' else False),
                                drop_last=True)
            
            val_subset = Subset(custom_dataset, val_idx)
            val_dl = DataLoader(val_subset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=(True if device == 'cuda' else False),
                                drop_last=True)
            
            for epoch in range(1, args.epochs+1):
                print(f'Epoch {epoch}/{args.epochs}')
                train(model, train_dl, device, criterion, sigmoid, optimizer, epoch, uniq_labels, verbose)
                f_measure = eval(model, val_dl, device, criterion, sigmoid, epoch, uniq_labels, verbose)
                metrics.append(f_measure)

                if epoch == args.epochs:
                    model_state_dict  = model.state_dict()
                    model_name = f'{epoch}-{f_measure:.4f}-split{i+1}'
                    save_model_torch(model_state_dict, model_name, args.model_folder)
                
        print('Averaged f-measure = ', round(np.mean(metrics), 3))
        torch.cuda.empty_cache()
        return None # Return None => All models are saved here already         

    else: 
        if verbose:
            print('Training only one model with full training data...')
        
        train_dl = DataLoader(custom_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=1,
                            pin_memory=(True if device == 'cuda' else False),
                            drop_last=True)
    
        for epoch in range(1, args.epochs+1):
            print(f'Epoch {epoch}/{args.epochs}')
            train(model, train_dl, device, criterion, sigmoid, optimizer, epoch, uniq_labels, verbose)
        
        torch.cuda.empty_cache()
        return model.state_dict()


def predict_proba(saved_model, data, classes, verbose, multi_dx_threshold=0.5):
    """
    NOTE: As opposed to the train function, this function takes in a signal 
    (numpy array) directly, instead of a path to a signal.

    Parameters:
        saved_model (pytorch state dict): trained model
        data (list): [signal (np.array), fs (int), age and gender features (np.array)]
        classes (list): List of possible unique labels
        verbose (bool): printouts?
        abnormal_threshold (float): threshold for additional multiclass labels

    Returns:
        probabilities (np.array): predicted probabilities for each class
    """

    # Consider the GPU or CPU condition
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load the test data
    test_set = CustomECGDataset(data)
    test_loader = DataLoader(test_set,
                         batch_size=1,
                         shuffle=False,
                         pin_memory=(True if device == 'cuda' else False),
                         drop_last=True)
    
    # Load the trained model
    model = ResNet(BasicBlock, [2, 2, 2, 2], 
                   in_channel=12, 
                   out_channel=len(classes))
    model.load_state_dict(saved_model)
    model.to(device)
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)

    # Run model on test data
    probabilities = test(model, test_loader, device, sigmoid, verbose)
    
    # Choose the class(es) with the highest probability as the label(s).
    # Set the threshold for additional labels here
    pred_dx = utils.multiclass_predict_from_logits(
                classes, probabilities, threshold=multi_dx_threshold
            )
    return pred_dx, probabilities