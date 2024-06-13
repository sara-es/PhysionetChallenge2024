import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold # For multilabel stratification

from classification.ResNet.SEResNet import ResNet, BasicBlock
from classification.ResNet.datasets.ECGDataset import ECGDataset, get_transforms
from utils import team_helper_code


def split_data(data, labels, n_splits=1):
    """
    Multilabel version
    Splitting data into two sets based on number of splits that are needed
    return indeces of the data for the splits
    """
    idx = np.arange(len(data))
    split_index_list = []

    if n_splits == 1: # One train/Test split
        mss = MultilabelStratifiedShuffleSplit(n_splits = n_splits, train_size=.75, test_size=.25, random_state=2024)
        for train_idx, test_idx in mss.split(idx, labels):
            split_index_list.append([train_idx, test_idx])
        
    else: # K-Fold
        skfold = MultilabelStratifiedKFold(n_splits = n_splits)
        for train_idx, test_idx in skfold.split(idx, labels):
            split_index_list.append([train_idx, test_idx])

    return split_index_list


def _split_data(data, labels, n_splits=1):  
    # Binary version
    idx = np.arange(len(data))
    split_indeces = []

    if n_splits == 1: # Basic train/test split
        train_idx, test_idx = train_test_split(idx, stratify=labels, test_size=.25, random_state=2024)
        split_indeces.append([train_idx, test_idx])
    
    else: # K-Fold
        skfold = StratifiedKFold(n_splits=n_splits)
        for train_idx, test_idx in skfold.split(idx, labels):
            split_indeces.append([train_idx, test_idx])

    return split_indeces


def train(model, train_loader, device, loss_fct, sigmoid, optimizer, epoch, uniq_labels, verbose):
    """
    
    """
    model.train()

    if verbose: # only need to keep track of this if we're printing it out
        running_loss = 0.0
        batches_per_printout = 200
        labels_all = torch.tensor((), device=device)
        logits_prob_all = torch.tensor((), device=device)

    with torch.set_grad_enabled(True):    
        for batch_idx, (ecgs, ag, labels) in enumerate(train_loader):
            ecgs = ecgs.float().to(device) # ECGs
            ag = ag.float().to(device) # age and gender
            labels = labels.float().to(device) # diagnoses in SNOMED CT codes 
            # TODO we should check the above - labels are not CT codes, but one-hot encoded corresponding to uniq_labels
        
            # Core training loop
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
                    f_measure = team_helper_code.compute_classification_metrics(
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
            ag = ag.float().to(device) # age and gender
            labels = labels.float().to(device) # diagnoses in SNOMED CT codes  

            # Run model
            logits = model(ecgs, ag) 

            # Calculate loss, append outputs to tensors
            epoch_loss += loss_fct(logits, labels)
            logits_prob_all = torch.cat((logits_prob_all, sigmoid(logits)), 0)  
            labels_all = torch.cat((labels_all, labels), 0)

    if verbose:
        epoch_loss = epoch_loss / len(train_loader.dataset)
        f_measure = team_helper_code.compute_classification_metrics(
                        labels_all, logits_prob_all, uniq_labels, threshold=0.5)
        print(f'Epoch {epoch}, val loss: {epoch_loss:.4f}, F-measure: {f_measure:.4f}')


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


def initialise_with_eval(train_data, train_labels, val_data, val_labels, device, batch_size=5):
    # Load the datasets       
    training_set = ECGDataset(train_data, get_transforms('train'), train_labels)
    train_dl = DataLoader(training_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1,
                          pin_memory=(True if device == 'cuda' else False),
                          drop_last=True)

    validation_set = ECGDataset(val_data, get_transforms('val'), val_labels) 
    validation_files = validation_set.data
    val_dl = DataLoader(validation_set,
                        batch_size=1,
                        shuffle=False,
                        num_workers=1,
                        pin_memory=(True if device == 'cuda' else False),
                        drop_last=True)
    
    return train_dl, val_dl


def initialise_train_only(train_data, train_labels, device, batch_size=5):
    training_set = ECGDataset(train_data, get_transforms('train'), train_labels)
    train_dl = DataLoader(training_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1,
                          pin_memory=(True if device == 'cuda' else False),
                          drop_last=True)
    
    return train_dl


def train_model(data, multilabels, uniq_labels, verbose, epochs=5, validate=True, n_splits=1):
    """
    Parameters:
        data (list): list of data where [path (str), fs (int), age and sex features (np.array)]
        multilabels (list): List of multilabels, one hot encoded
        uniq_labels (list): List of unique labels as strings
        verbose (bool): printouts?
        epochs (int): number of epochs to train
        validate (bool): perform validation?
    Returns:
        state_dict (pytorch model): state dictionary of the trained model
        metrics (float): F-measure (if validate=True, else None)
    """
    # channels is just hard coded for now
    # could also set channels = len(sig[1]['units']) where sig = helper_code.load_signal(record)

    # # Consider the GPU or CPU condition
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print('Using gpu(s)')
    else:
        device = torch.device("cpu")
        if verbose:
            print('Using cpu')

    model = ResNet(BasicBlock, [2, 2, 2, 2], 
                in_channel=12, 
                out_channel=len(uniq_labels))
    # Optimizer
    optimizer = optim.Adam(model.parameters(), 
                            lr=0.003,
                            weight_decay=0.00001)
    
    criterion = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)
    model.to(device)
    
    if validate:
        # Split data to training and validation; return indices for training and validation sets
        # Either one stratified train/val split OR Stratified K-fold
        # Default: one train/val split
        split_index_list = split_data(data, multilabels, n_splits=n_splits) 
        for train_idx, val_idx in split_index_list:
            train_data, val_data = list(map(data.__getitem__, train_idx)), list(map(data.__getitem__, val_idx))
            train_labels, val_labels = list(map(multilabels.__getitem__, train_idx)), list(map(multilabels.__getitem__, val_idx))
            # Iterate over train/test splits
            train_dl, val_dl = initialise_with_eval(train_data, train_labels, val_data, val_labels, device, batch_size=5)
            
            # Training ResNet model(s) on the training data and evaluating on the validation set
            # Need to include unique labels here for F-measure calculation
            for epoch in range(1, epochs+1):
                train(model, train_dl, device, criterion, sigmoid, optimizer, epoch, uniq_labels, verbose)
                eval(model, val_dl, device, criterion, sigmoid, epoch, uniq_labels, verbose)
    
    else: 
        # Only train the model
        # Train the model using entire data and store the state dictionary
        train_dl = initialise_train_only(data, multilabels, device, batch_size=5)
        
        for epoch in range(1, epochs+1):
            train(model, train_dl, device, criterion, sigmoid, optimizer, epoch, uniq_labels, verbose)

    return model.state_dict()


def predict_proba(saved_model, data, classes, verbose, abnormal_threshold=0.5):
    """
    NOTE: As opposed to the train function, this function takes in a signal 
    (numpy array) directly, instead of a path to a signal.

    Parameters:
        saved_model (pytorch state dict): trained model
        data (list): [signal (np.array), fs (int), age and gender features (np.array)]
        classes (list): List of possible unique labels
        verbose (bool): printouts?
        abnormal_threshold (float): threshold for 'Abnormal' class
            (Used to be multi_dx_threshold for additional multiclass labels)

    Returns:
        probabilities (np.array): predicted probabilities for each class
    """

    # Consider the GPU or CPU condition
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load the test data
    test_set = ECGDataset(data, get_transforms('test'))
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
    pred_dx = team_helper_code.threshold_predict_from_logits(
                classes, probabilities, threshold=abnormal_threshold
            )
    return pred_dx, probabilities