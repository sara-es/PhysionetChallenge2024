import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold # For multilabel stratification

from classification.SEResNet import resnet18
from classification.ECGDataset import ECGDataset, get_transforms
from utils.team_helper_code import compute_classification_metrics


class Training(object):
    def __init__(self, args):
        self.args = args
  
    def setup(self):
        '''Initializing the device conditions, datasets, dataloaders, 
        model, loss, criterion and optimizer
        '''

        # Consider the GPU or CPU condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = 1
            if self.args['verbose']:
                print('using {} gpu(s)'.format(self.device_count))
            assert self.args['batch_size'] % self.device_count == 0, "batch size should be divided by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            if self.args['verbose']:
                print('using {} cpu'.format(self.device_count))

        # Load the datasets       
        training_set = ECGDataset(self.args['train_data'], 
                                  get_transforms('train'),
                                  self.args['train_labels'],
                                  )
        channels = training_set.channels
        self.train_dl = DataLoader(training_set,
                                   batch_size=self.args['batch_size'],
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=(True if self.device == 'cuda' else False),
                                   drop_last=True)

        if self.args['val_data'] is not None:
            validation_set = ECGDataset(self.args['val_data'],
                                        get_transforms('val'),
                                        self.args['val_labels'], 
                                        ) 
            self.validation_files = validation_set.data
            self.val_dl = DataLoader(validation_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=1,
                                    pin_memory=(True if self.device == 'cuda' else False),
                                    drop_last=True)

        self.model = resnet18(in_channel=channels, 
                              out_channel=len(self.args['dx_labels']))

        # If more than 1 CUDA device used, use data parallelism
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model) 
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=0.003,
                                    weight_decay= 0.00001)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)
        
    def train(self, compute_metrics=False):
        ''' PyTorch training loop
        '''

        if self.args['verbose']:
                print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s' % \
                    (type(self.model).__name__, 
                    type(self.optimizer).__name__,
                    self.optimizer.param_groups[0]['lr'], 
                    self.args['epochs'], 
                    self.device))
                
        f_measure = None

        for epoch in range(1, self.args['epochs']):
            
            # --- TRAIN ON TRAINING SET -----------------------------
            self.model.train()            
            train_loss = 0.0
            labels_all = torch.tensor((), device=self.device) # , device=torch.device('cuda:0')
            logits_prob_all = torch.tensor((), device=self.device)
            
            batch_loss = 0.0
            batch_count = 0
            step = 0
            
            for batch_idx, (ecgs, ag, labels) in enumerate(self.train_dl):
                ecgs = ecgs.float().to(self.device) # ECGs
                ag = ag.float().to(self.device) # age and gender
                labels = labels.float().to(self.device) # diagnoses in SNOMED CT codes  
               
                with torch.set_grad_enabled(True):                    
        
                    logits = self.model(ecgs, ag) 
                    loss = self.criterion(logits, labels)
                    logits_prob = self.sigmoid(logits)      
                    loss_tmp = loss.item() * ecgs.size(0)
                    labels_all = torch.cat((labels_all, labels), 0)
                    logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)                    
                    
                    train_loss += loss_tmp
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Printing training information
                    if step % 100 == 0:
                        batch_loss += loss_tmp
                        batch_count += ecgs.size(0)
                        batch_loss = batch_loss / batch_count
                        if self.args['verbose']:
                            print('epoch {:^3} [{}/{}] train loss: {:>5.4f}'.format(
                                epoch, 
                                batch_idx * len(ecgs), 
                                len(self.train_dl.dataset), 
                                batch_loss
                            ))

                        batch_loss = 0.0
                        batch_count = 0
                    step += 1

            train_loss = train_loss / len(self.train_dl.dataset)            

            if self.args['val_data'] is not None:
            # --- EVALUATE ON VALIDATION SET ------------------------------------- 
                self.model.eval()
                val_loss = 0.0  
                labels_all = torch.tensor((), device=self.device)
                logits_prob_all = torch.tensor((), device=self.device)  
                threshold = 0.5
                
                for ecgs, ag, labels in self.val_dl:
                    ecgs = ecgs.float().to(self.device) # ECGs
                    ag = ag.float().to(self.device) # age and gender
                    labels = labels.float().to(self.device) # diagnoses in SNOMED CT codes 
                    
                    with torch.set_grad_enabled(False):  
                        
                        logits = self.model(ecgs, ag)
                        loss = self.criterion(logits, labels)
                        logits_prob = self.sigmoid(logits)
                        val_loss += loss.item() * ecgs.size(0)                                 
                        labels_all = torch.cat((labels_all, labels), 0)
                        logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

                val_loss = val_loss / len(self.val_dl.dataset)

                # Return metrics when validating
                f_measure = compute_classification_metrics(labels_all, logits_prob_all, threshold)

        model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
        return model_state_dict, f_measure
    
 
class Predicting(object):
    def __init__(self, args):
        self.args = args
    
    def setup(self):
        ''' Initializing the device conditions and dataloader,
        loading trained model
        '''
        # Consider the GPU or CPU condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
        
        # Load the test data
        testing_set = ECGDataset(self.args['test_data'], 
                                 get_transforms('test'))
        channels = testing_set.channels
        self.test_dl = DataLoader(testing_set,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=(True if self.device == 'cuda' else False),
                                  drop_last=True)
        
        # Load the trained model
        self.model = resnet18(in_channel=channels, 
                              out_channel=len(self.args['dx_labels']))
        self.model.load_state_dict(self.args['model'])

        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)
        
    def predict(self):
        ''' Make predictions
        '''
        # if self.args['verbose']: 
        #     print('predict() called: model={}, device={}'.format(
        #         type(self.model).__name__,
        #         self.device))
 
        # --- EVALUATE ON TESTING SET ------------------------------------- 
        self.model.eval()
        logits_prob_all = torch.tensor((), device=self.device)  
        
        for i, (ecgs, ag) in enumerate(self.test_dl):
            ecgs = ecgs.float().to(self.device) # ECGs
            ag = ag.float().to(self.device) # age and gender

            with torch.set_grad_enabled(False):  
                logits = self.model(ecgs, ag)
                logits_prob = self.sigmoid(logits)
                logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)
           
            return logits_prob.cpu().detach().numpy().squeeze()
            
        torch.cuda.empty_cache()


# =============== Main train and test function calls =============================
# Multilabel version
# Splitting data into two sets based on number of splits that are needed
# return indeces of the data for the splits
def split_data(data, labels, n_splits=1):
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

# Binary version
def _split_data(data, labels, n_splits=1):
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


def train(data, multilabels, uniq_labels, verbose, epochs=5, validate=True):
    """
    Parameters:
        data (list): list of data where [path (str), fs (int), age and sex features (np.array)]
        multilabels (list): List of multilabels
        uniq_labels (list): List of unique labels
        verbose (bool): printouts?
        epochs (int): number of epochs to train
        validate (bool): perform validation?
    Returns:
        state_dict (pytorch model): state dictionary of the trained model
        metrics (float): F-measure (if validate=True, else None)
    """
    # n channels is now set in the ECGDataset class so no need to set it here unless we choose otherwise :)
    # can also set channels = len(sig[1]['units']) where sig = helper_code.load_signal(record)
    
    # ============= VALIDATION? =============
    if validate:
        # 1) Split data to training and validation; return indeces for training and validation sets
        # Either one stratified train/val split OR Stratified K-fold
        split_index_list = split_data(data, multilabels, n_splits=1) # Default, one train/val split

        # Iterate over train/test splits
        pool_metrics = []
        for train_idx, val_idx in split_index_list:
            train_data, val_data = list(map(data.__getitem__, train_idx)), list(map(data.__getitem__, val_idx))
            train_labels, val_labels = list(map(multilabels.__getitem__, train_idx)), list(map(multilabels.__getitem__, val_idx))

            args = {'train_data': train_data, 'val_data': val_data,
                    'train_labels': train_labels, 'val_labels': val_labels,
                    'dx_labels': uniq_labels, 'epochs': epochs, 'batch_size': 5,
                    'verbose': verbose}
            
            # 2) Training ResNet model(s) on the training data and evaluating on the validation set
            trainer = Training(args)
            trainer.setup()
            state_dict, metrics = trainer.train(compute_metrics=True) # Compute also the classification metrics (now, F-measure)
            pool_metrics.append(metrics)  

        if verbose:
            print('\nValidation phase performed using {}'.format('basic train/val split' 
                                                                if len(split_index_list) == 1 
                                                                else '{}-Fold'.format(len(split_index_list ))))
            print('\t - F-measure: {}'.format(pool_metrics[0] 
                                            if len(split_index_list) == 1 
                                            else np.nanmean(pool_metrics)))
    
    else: # Only train the model
        # Train the model using entire data and store the state dictionary
        args = {'train_data': data, 'val_data': None,
                'train_labels': multilabels, 'val_labels': None,
                'dx_labels': uniq_labels, 'epochs': epochs, 'batch_size': 5,
                'verbose': verbose}
        
        trainer = Training(args)
        trainer.setup()
        state_dict, _ = trainer.train() 

    return state_dict


def predict_proba(model, data, classes, verbose):
    """
    Parameters:
        model (pytorch model): trained model
        data (list): [path (str), fs (int), age and gender features (np.array)]
        classes (list): List of possible unique labels
        verbose (bool): printouts?

    Returns:
        probabilities (np.array): predicted probabilities for each class
    """
    # Predict the probabilities for the classes
    args = {'model': model, 'test_data': data, 'dx_labels': classes, 'threshold': 0.5, 
            'device_count': 1, 'verbose': verbose}
    
    predictor = Predicting(args)
    predictor.setup()
    probabilities = predictor.predict()

    return probabilities