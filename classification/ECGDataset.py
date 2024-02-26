import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import torch
from torch.utils.data import Dataset
from preprocessing.transforms import Compose, RandomClip, Normalize, ValClip, Retype
from helper_code import load_signal

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
    
    :param path: Paths of the filenames for the records 
    :type path: str
    :param preprocess: Preprocess transforms for ECG recording
    :type preprocess: datasets.transforms.Compose
    :param transform: The other transforms used for ECG recording
    :type transform: datasets.transforms.Compose
    '''

    def __init__(self, records, labels, features, fss, transforms):
        self.data = records
        self.labels = labels
        self.ag = features
        self.fs = fss
        self.transforms = transforms
        self.channels = 12
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file_name = self.data[item]
        fs = self.fs[item]
        ecg, _ = load_signal(file_name)
        ecg = ecg.T # CHECK THIS OUT!?!?!!?

        label = self.labels[item]
        ag = self.ag[item]
        return ecg, torch.from_numpy(ag).float(), torch.from_numpy(label).float()