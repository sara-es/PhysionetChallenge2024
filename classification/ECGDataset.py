import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import torch
from torch.utils.data import Dataset
from preprocessing.transforms import Compose, RandomClip, Normalize, ValClip, Retype
from helper_code import load_signal
import numpy as np


def get_transforms(dataset_type):
    ''' Get transforms for ECG data based on the dataset type (train, validation, test)
    '''
    seq_length = 4096
    normalizetype = '0-1'
    
    data_transforms = {
        
        'train': Compose([
            RandomClip(w=seq_length),
            Normalize(normalizetype),
            Retype() 
        ], p = 1.0),
        
        'val': Compose([
            ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0),
        
        'test': Compose([
            ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0)
    }
    return data_transforms[dataset_type]


class ECGDataset(Dataset):
    ''' Class implementation of Dataset of ECG recordings
    
    :param data: list of data where [path (str), fs (int), age and sex features (np.array)]
    :type data: lst[lst]
    :param labels: List of multilabels
    :type labels: lst[lst]
    :param transform: The transforms to use for ECG recordings
    :type transform: datasets.transforms.Compose
    '''

    def __init__(self, data, transforms, labels=None):
        self.data = [ls[0] for ls in data] # ECG paths, str
        self.fs = [ls[1] for ls in data] # fs, int
        self.ag = [ls[2] for ls in data] # 89-year-old Male => [0.89, 0, 1]
        self.transforms = transforms
        self.channels = 12
        if labels is not None:
            self.labels = labels
            self.training = True
        else:
            self.training = False
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file = self.data[item]
        if type(file) == np.ndarray: # in training, the signal is passed in directly
            ecg = file
        else:
            ecg, _ = load_signal(file) # shape (samples, channels)
        fs = self.fs[item]
        ecg = ecg.T # shape (channels, samples)
        ag = self.ag[item]

        if self.training:
            label_torch = self.labels[item]
            return ecg, torch.from_numpy(ag).float(), torch.from_numpy(label_torch).float()
        else:
            return ecg, torch.from_numpy(ag).float()