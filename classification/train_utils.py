import os, sys
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from seresnet18 import resnet18
from ECGDataset import ECGDataset, get_transforms
import pickle

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
            self.device_count = self.args.device_count
            print('using {} gpu(s)'.format(self.device_count))
            assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            print('using {} cpu'.format(self.device_count))

        # Load the datasets       
        training_set = ECGDataset(self.args.train_path, get_transforms('train'))
        validation_set = ECGDataset(self.args.val_path, get_transforms('val')) 
        channels = training_set.channels
        self.validation_files = validation_set.data
              
        self.train_dl = DataLoader(training_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   num_workers=self.args.num_workers,
                                   pin_memory=(True if self.device == 'cuda' else False),
                                   drop_last=True)
        
        self.val_dl = DataLoader(validation_set,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.args.num_workers,
                                 pin_memory=(True if self.device == 'cuda' else False),
                                 drop_last=True)

        self.model = resnet18(in_channel=channels, 
                              out_channel=len(self.args.labels))

        # If more than 1 CUDA device used, use data parallelism
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model) 
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)
        
    def train(self):
        ''' PyTorch training loop
        '''
        
        print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
              (type(self.model).__name__, 
               type(self.optimizer).__name__,
               self.optimizer.param_groups[0]['lr'], 
               self.args.epochs, 
               self.device))
        
        for epoch in range(1, self.args.epochs+1):
            
            # --- TRAIN ON TRAINING SET -----------------------------
            self.model.train()            
            train_loss = 0.0
            labels_all = torch.tensor((), device=self.device) # , device=torch.device('cuda:0')
            logits_prob_all = torch.tensor((), device=self.device)
            
            batch_loss = 0.0
            batch_count = 0
            step = 0
            
            for batch_idx, (ecgs, ag, labels) in enumerate(self.train_dl):
                ecgs = ecgs.to(self.device) # ECGs
                ag = ag.to(self.device) # age and gender
                labels = labels.to(self.device) # diagnoses in SNOMED CT codes  
               
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

            # --- EVALUATE ON VALIDATION SET ------------------------------------- 
            self.model.eval()
            val_loss = 0.0  
            labels_all = torch.tensor((), device=self.device)
            logits_prob_all = torch.tensor((), device=self.device)  
            
            for ecgs, ag, labels in self.val_dl:
                ecgs = ecgs.to(self.device) # ECGs
                ag = ag.to(self.device) # age and gender
                labels = labels.to(self.device) # diagnoses in SNOMED CT codes 
                
                with torch.set_grad_enabled(False):  
                    
                    logits = self.model(ecgs, ag)
                    loss = self.criterion(logits, labels)
                    logits_prob = self.sigmoid(logits)
                    val_loss += loss.item() * ecgs.size(0)                                 
                    labels_all = torch.cat((labels_all, labels), 0)
                    logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

            val_loss = val_loss / len(self.val_dl.dataset)

        model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
        return model_state_dict
