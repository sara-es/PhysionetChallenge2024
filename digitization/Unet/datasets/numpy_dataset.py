# Nicola Dinsdale 2020
# Pytorch dataset for numpy arrays
########################################################################################################################
# Import dependencies
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
                options = ['jitter', 'blur', 'noise', 'rotation']
                choice = torch.randint(low=0, high=4, size=(1,))
                option = options[choice]
                if option == 'jitter':
                    jitter = transforms.ColorJitter(brightness=.5, hue=.3)
                    x = jitter(x)
                elif option == 'rotation':
                    rot = int(torch.randint(low=0, high=90, size=(1,)))
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
        return len(self.data)
    



    