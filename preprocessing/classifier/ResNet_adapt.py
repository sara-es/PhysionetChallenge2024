from torchvision.models import resnet18
import torch.nn as nn 

class ResNet_adapt(nn.Module):

    def __init__(self, in_channels=1, embsize=128,  weights=None):
        super(ResNet_adapt, self).__init__()

        # preload resnet
        self.model = resnet18(weights=weights)
        #self.model.conv1 =  nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc =  nn.Linear(in_features=512, out_features=embsize, bias=True)

    def forward(self, x):
        return self.model(x)
    