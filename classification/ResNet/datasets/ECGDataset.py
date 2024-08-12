import sys, os, numpy as np
sys.path.append(os.path.join(sys.path[0], '..'))

import torch
from torch.utils.data import Dataset
from classification.ResNet.datasets.transforms import Compose, SplineInterpolation, Normalize, ValClip, Retype, DigitizationClip, DigitizationClipMissing
from helper_code import load_signals
import numpy as np


def get_transforms(dataset_type, fs):
    ''' Get transforms for ECG data based on the dataset type (train, validation, test)
    '''
    # TODO: change seq_length to be more than 10 seconds
    seq_length = 1024 
    normalizetype = '0-1'
    new_fs = 100
    
    data_transforms = {
        'train': Compose([
            #RandomClip(w=seq_length),
            SplineInterpolation(fs_new=new_fs, fs_old=fs),
            DigitizationClip(w=seq_length, fs=new_fs),
            Normalize(normalizetype),
            Retype() 
        ], p = 1.0),
        'val': Compose([
            #TODO: check whether DigitizationClip is needed?
            SplineInterpolation(fs_new=new_fs, fs_old=fs),
            DigitizationClip(w=seq_length, fs=new_fs),
            #ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0),
        # no need for crops, as this should take in the result of digitization step
        'test': Compose([
            #ValClip(w=seq_length),
            SplineInterpolation(fs_new=new_fs, fs_old=fs),
            DigitizationClip(w=seq_length, fs=new_fs),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0)
    }
    return data_transforms[dataset_type]

def get_transforms_missing(dataset_type, fs):
    ''' Get transforms for ECG data based on the dataset type (train, validation, test)
    '''
    # TODO: change seq_length to be more than 10 seconds
    seq_length = 1024 
    normalizetype = '0-1'
    new_fs = 100
    
    data_transforms = {
        'train': Compose([
            #RandomClip(w=seq_length),
            SplineInterpolation(fs_new=new_fs, fs_old=fs),
            DigitizationClipMissing(w=seq_length, fs=new_fs),
            Normalize(normalizetype),
            Retype() 
        ], p = 1.0),
        'val': Compose([
            #TODO: check whether DigitizationClip is needed?
            SplineInterpolation(fs_new=new_fs, fs_old=fs),
            DigitizationClipMissing(w=seq_length, fs=new_fs),
            #ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0),
        # no need for crops, as this should take in the result of digitization step
        'test': Compose([
            #ValClip(w=seq_length),
            SplineInterpolation(fs_new=new_fs, fs_old=fs),
            DigitizationClipMissing(w=seq_length, fs=new_fs),
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
        self.ag = [ls[2] for ls in data] # 89-year-old Male => [0.89, 0, 1, 1, 1] with missing data flags
        self.transforms = [get_transforms(transforms, fs) for fs in self.fs]
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

        ecg = self.transforms[item](ecg)

        if self.training:
            label_torch = self.labels[item]
            return ecg, torch.from_numpy(ag).float(), torch.from_numpy(label_torch).float()
        else:
            return ecg, torch.from_numpy(ag).float()
        
class ECGDataset_missing(Dataset):
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
        self.ag = [ls[2] for ls in data] # 89-year-old Male => [0.89, 0, 1, 1, 1] with missing data flags
        self.transforms = [get_transforms_missing(transforms, fs) for fs in self.fs]
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

        ecg = self.transforms[item](ecg)

        if self.training:
            label_torch = self.labels[item]
            return ecg, torch.from_numpy(ag).float(), torch.from_numpy(label_torch).float()
        else:
            return ecg, torch.from_numpy(ag).float()
