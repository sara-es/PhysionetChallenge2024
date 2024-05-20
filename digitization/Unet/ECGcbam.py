import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, active=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=int((kernel_size - 1) / 2),
            bias=False
        )

        self.active = active
        self.activate = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        return

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        if self.active:
            out = self.activate(out)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=8):
        super(ChannelGate, self).__init__()

        pro_channels = gate_channels // reduction_ratio

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, pro_channels),
            nn.ReLU(inplace=True),
            nn.Linear(pro_channels, gate_channels)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, x):
        avgpool = self.avgpool(x)
        maxpool = self.maxpool(x)
        channel_att_sum = self.mlp(avgpool) + self.mlp(maxpool)

        scale = self.sigmoid(channel_att_sum)
        scale = scale.unsqueeze(2).unsqueeze(3)
        scale = scale.expand_as(x)
        return x * scale


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = ConvBlock(2, 1, 7, active=False)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, x):
        avgmap = torch.mean(x, 1).unsqueeze(1)
        maxmap = torch.max(x, 1)[0].unsqueeze(1)
        compress = torch.cat([avgmap, maxmap], dim=1)

        scale = self.spatial(compress)
        scale = self.sigmoid(scale)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=8, apply_spatial=True):
        super(CBAM, self).__init__()

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate() if apply_spatial else None
        return

    def forward(self, x):
        out = self.ChannelGate(x)
        if self.SpatialGate is not None:
            out = self.SpatialGate(out)
        return out


if __name__ == '__main__':

    tensor = torch.rand((16, 64, 128, 128))
    cbam = CBAM(64, 16, True)
    out = cbam(tensor)
    print(out.size())
