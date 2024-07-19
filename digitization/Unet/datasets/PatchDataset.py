import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import torchio as tio

class PatchDataset(Dataset):  # Inherit from Dataset class
    def __init__(self, ids, im_patch_dir, label_patch_dir, train=True, transform=None):
        self.ids = ids
        self.im_patch_dir = im_patch_dir
        self.label_patch_dir = label_patch_dir
        self.train = train
        # self.data = torch.from_numpy(data).float()
        # self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        im_path = os.path.join(self.im_patch_dir, self.ids[index])
        with open(im_path, 'rb') as f:
            im = np.load(f)
            # reshape to (c, w, h)
            x = torch.from_numpy(np.transpose(im, (2, 0, 1))).float()
        
        if self.train:
            lab_path = os.path.join(self.label_patch_dir, self.ids[index])
            with open(lab_path, 'rb') as f:
                lab = np.load(f)
                # make the labels one hot
                y_store = np.zeros((2, lab.shape[0], lab.shape[1]))
                y_store[0][lab==0] = 1
                y_store[1][lab==1] = 1
                y = torch.from_numpy(y_store).float()
        else:
            y = torch.zeros(2, x.shape[1], x.shape[2])

        if self.transform:
            val = torch.rand(1)
            if val > 0.5:   # Augment with probability 0.5
                options = ['jitter', 'blur', 'noise', 'rotation']
                choice = torch.randint(low=0, high=4, size=(1,))
                option = options[choice]
                if option == 'jitter':
                    jitter = transforms.ColorJitter(brightness=.5, hue=.3)
                    x = jitter(x)
                elif option == 'rotation':
                    rot = int(torch.randint(low=-90, high=90, size=(1,)))
                    x = TF.rotate(x, rot)
                    y = TF.rotate(y, rot)
                elif option == 'blur':
                    blurer = transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.5, 0.5))
                    x = blurer(x)
                elif option == 'noise':
                    noise = torch.randn(x.shape)
                    x = x + noise
        return x, y

    def __len__(self):
        return len(self.ids)
    