import sys, os, numpy as np
sys.path.append(os.path.join(sys.path[0], '..'))

import torch
from torch.utils.data import Dataset
from classification.ResNet.datasets.transforms import Compose, RandomClip, Normalize, ValClip, Retype, DigitizationClip
from helper_code import load_signals
import numpy as np


def get_transforms(dataset_type):
    ''' Get transforms for ECG data based on the dataset type (train, validation, test)
    '''
    # TODO: change seq_length to be more than 10 seconds
    #seq_length = 4096
    seq_length = 5000
    normalizetype = '0-1'
    
    data_transforms = {
        'train': Compose([
            #RandomClip(w=seq_length),
            DigitizationClip(w=seq_length),
            Normalize(normalizetype),
            Retype() 
        ], p = 1.0),
        'val': Compose([
            #TODO: check whether DigitizationClip is needed?
            DigitizationClip(w=seq_length),
            #ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0),
        # no need for crops, as this should take in the result of digitization step
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
            ecg, _ = load_signals(file) # shape (samples, channels)
        fs = self.fs[item]
        ecg = ecg.T # shape (channels, samples)
        ag = self.ag[item]

        ecg = np.nan_to_num(ecg, nan=0) # HANDLING THE NAN VALUES IN SIGNAL DATA

        if self.training:
            label_torch = self.labels[item]
            return ecg, torch.from_numpy(ag).float(), torch.from_numpy(label_torch).float()
        else:
            return ecg, torch.from_numpy(ag).float()
