import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import torchio as tio
########################################################################################################################

class numpy_dataset(Dataset):  # Inherit from Dataset class
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            val = torch.rand(1)
            if val > 0.5:   # Augment with probability 0.5
                options = ['jitter', 'blur', 'noise', 'rotation', 'scale']
                choice = torch.randint(low=0, high=len(options), size=(1,))
                option = options[choice]
                if option == 'jitter':
                    jitter = transforms.ColorJitter(brightness=.2, hue=.2)
                    x = jitter(x)
                elif option == 'rotation':
                    rot = int(torch.randint(low=0, high=90, size=(1,)))
                    x = TF.rotate(x, rot)
                elif option == 'blur':
                    blurer = transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.5, 0.5))
                    x = blurer(x)
                elif option == 'noise':
                    noise = torch.randn(x.shape)
                    x = x + noise
                elif option == 'scale':
                    scaler = transforms.RandomAffine(degrees=0, translate=None, scale=(0.5, 1.5))     # Nearest neighbour interpolation by standard, use that for now 
                    x = scaler(x)
        return x, y

    def __len__(self):
        return len(self.data)
    

class PatchDataset(Dataset):  # Inherit from Dataset class
    def __init__(self, im_patch_path, label, train=True, transform=False):
        self.id = im_patch_path
        self.target = label
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        # im_path = os.path.join(self.im_patch_dir, self.ids[index])
        with open(self.id[index], 'rb') as f:
            im = np.load(f)
        # reshape to (c, w, h)
        x_store = np.transpose(im, (2, 0, 1))
        x_store = (x_store - x_store.mean())/x_store.std()
        x = torch.from_numpy(x_store).float()
        
        if self.train:
            y_store = np.zeros(2)
            y_store[self.target[index]] = 1.
            y = torch.from_numpy(y_store).float()
        else:
            y = torch.zeros(1)

        if self.transform:
            val = torch.rand(1)
            if val > 0.5:   # Augment with probability 0.5
                options = ['jitter', 'blur', 'noise', 'rotation', 'scale']
                choice = torch.randint(low=0, high=len(options), size=(1,))
                option = options[choice]
                if option == 'jitter':
                    jitter = transforms.ColorJitter(brightness=.2, hue=.2)
                    x = jitter(x)
                elif option == 'rotation':
                    rot = int(torch.randint(low=0, high=90, size=(1,)))
                    x = TF.rotate(x, rot)
                elif option == 'blur':
                    blurer = transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.5, 0.5))
                    x = blurer(x)
                elif option == 'noise':
                    noise = torch.randn(x.shape)
                    x = x + noise
                elif option == 'scale':
                    scale = float(torch.rand(1)) + 0.5
                    scaler = transforms.RandomAffine(degrees=0, translate=None, scale=(scale, scale), fill=1) # Nearest neighbour interpolation by standard, use that for now 
                    x = scaler(x)
        return x, y

    def __len__(self):
        return len(self.id)
    



    