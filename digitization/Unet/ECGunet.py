import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import torch
import torch.nn as nn

from digitization.Unet.ECGcbam import CBAM

# code from https://www.sciencedirect.com/science/article/pii/S001048252030408X?via%3Dihub#appsec1

class ConvNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1, bias=False):
        super(ConvNBlock, self).__init__()

        self.convn = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=int((kernel_size - 1) / 2),
            groups=groups, bias=bias
        )
        return

    def forward(self, x):
        out = self.convn(x)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, cbam=False, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvNBlock(in_channels, out_channels, 3, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = ConvNBlock(out_channels, out_channels, 3, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.cbam = CBAM(out_channels, 8, False) if cbam else None

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ConvNBlock(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        return

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += identity
        out = self.act2(out)
        return out


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, cbam=False, stride=1,
                 groups=1, expansion=4, base_channels=64,):
        super(BottleneckBlock, self).__init__()

        group_channels = out_channels * base_channels / 64
        pro_channels = int(group_channels) * groups
        out_channels *= expansion

        self.conv1 = ConvNBlock(in_channels, pro_channels, 1, 1, 1)
        self.bn1 = nn.BatchNorm2d(pro_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = ConvNBlock(pro_channels, pro_channels, 3, stride, groups)
        self.bn2 = nn.BatchNorm2d(pro_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = ConvNBlock(pro_channels, out_channels, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.cbam = CBAM(out_channels, 8, False) if cbam else None
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ConvNBlock(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        return

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activate(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += identity
        out = self.activate(out)
        return out


class BasicEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, cbam=False, stride=1):
        super(BasicEncoder, self).__init__()

        layers = [BasicBlock(in_channels, out_channels, cbam, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, cbam, 1))

        self.encoder = nn.Sequential(*layers)
        return

    def forward(self, x):
        out = self.encoder(x)
        return out


class BottleneckEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, cbam=False, stride=1,
                 groups=1, expansion=4, base_channels=64):
        super(BottleneckEncoder, self).__init__()

        layers = [BottleneckBlock(in_channels, out_channels, cbam, stride,
                                  groups, expansion, base_channels)]
        in_channels = out_channels * expansion
        for _ in range(1, n_blocks):
            layers.append(BottleneckBlock(in_channels, out_channels, cbam, 1,
                                          groups, expansion, base_channels))

        self.encoder = nn.Sequential(*layers)
        return

    def forward(self, x):
        out = self.encoder(x)
        return out


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks=1, cbam=False):
        super(DecoderBlock, self).__init__()

        layers = [BasicBlock(in_channels, out_channels, cbam, 1)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, cbam, 1))

        self.decoder = nn.Sequential(*layers)
        return

    def forward(self, x):
        out = self.decoder(x)
        return out


class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, factor=2):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True),
            ConvNBlock(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return

    def forward(self, x):
        out = self.upconv(x)
        return out


class BasicResUNet(nn.Module):

    def __init__(self, in_channels, out_channels, nbs=[1, 1, 1, 1],
                 init_channels=16, cbam=False):
        super(BasicResUNet, self).__init__()

        # number of kernels
        nks = [init_channels * (2 ** i) for i in range(0, len(nbs) + 1)]

        self.input = nn.Sequential(
            ConvNBlock(in_channels, nks[0], 7),
            nn.BatchNorm2d(nks[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.econv1 = BasicEncoder(nks[0], nks[1], nbs[0], cbam, 2)
        self.econv2 = BasicEncoder(nks[1], nks[2], nbs[1], cbam, 2)
        self.econv3 = BasicEncoder(nks[2], nks[3], nbs[2], cbam, 2)
        self.econv4 = BasicEncoder(nks[3], nks[4], nbs[3], cbam, 2)

        self.uconv4 = UpConvBlock(nks[4], nks[3], 2)
        self.uconv3 = UpConvBlock(nks[3], nks[2], 2)
        self.uconv2 = UpConvBlock(nks[2], nks[1], 2)
        self.uconv1 = UpConvBlock(nks[1], nks[0], 2)

        self.dconv4 = DecoderBlock(nks[3] * 2, nks[3], 1, cbam)
        self.dconv3 = DecoderBlock(nks[2] * 2, nks[2], 1, cbam)
        self.dconv2 = DecoderBlock(nks[1] * 2, nks[1], 1, cbam)
        self.dconv1 = DecoderBlock(nks[0] * 2, nks[0], 1, cbam)

        self.output = ConvNBlock(nks[0], out_channels, 1)
        return

    def forward(self, x):
        x = self.input(x)
        e1 = self.econv1(x)
        e2 = self.econv2(e1)
        e3 = self.econv3(e2)
        e4 = self.econv4(e3)

        u4 = self.uconv4(e4)
        c4 = torch.cat([e3, u4], dim=1)
        d4 = self.dconv4(c4)

        u3 = self.uconv3(d4)
        c3 = torch.cat([e2, u3], dim=1)
        d3 = self.dconv3(c3)

        u2 = self.uconv2(d3)
        c2 = torch.cat([e1, u2], dim=1)
        d2 = self.dconv2(c2)

        u1 = self.uconv1(d2)
        c1 = torch.cat([x, u1], dim=1)
        d1 = self.dconv1(c1)

        out = self.output(d1)
        return out


class BottleneckResUNet(nn.Module):

    def __init__(self, in_channels, out_channels, nbs=[1, 1, 1, 1],
                 init_channels=16, cbam=False):
        super(BottleneckResUNet, self).__init__()

        # number of kernels
        nks = [init_channels * (2 ** i) for i in range(0, len(nbs) + 1)]

        self.input = nn.Sequential(
            ConvNBlock(in_channels, nks[1], 7),
            nn.BatchNorm2d(nks[1]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.econv1 = BottleneckEncoder(nks[1], nks[0], nbs[0], cbam, 2)
        self.econv2 = BottleneckEncoder(nks[2], nks[1], nbs[1], cbam, 2)
        self.econv3 = BottleneckEncoder(nks[3], nks[2], nbs[2], cbam, 2)
        self.econv4 = BottleneckEncoder(nks[4], nks[3], nbs[3], cbam, 2)

        self.uconv4 = UpConvBlock(nks[4] * 2, nks[4], 2)
        self.uconv3 = UpConvBlock(nks[3] * 2, nks[3], 2)
        self.uconv2 = UpConvBlock(nks[2] * 2, nks[2], 2)
        self.uconv1 = UpConvBlock(nks[1] * 2, nks[1], 2)

        self.dconv4 = DecoderBlock(nks[4] * 2, nks[4], 1, cbam)
        self.dconv3 = DecoderBlock(nks[3] * 2, nks[3], 1, cbam)
        self.dconv2 = DecoderBlock(nks[2] * 2, nks[2], 1, cbam)
        self.dconv1 = DecoderBlock(nks[1] * 2, nks[1], 1, cbam)

        self.output = ConvNBlock(nks[1], out_channels, 1)
        return

    def forward(self, x):
        x = self.input(x)
        e1 = self.econv1(x)
        e2 = self.econv2(e1)
        e3 = self.econv3(e2)
        e4 = self.econv4(e3)

        u4 = self.uconv4(e4)
        c4 = torch.cat([e3, u4], dim=1)
        d4 = self.dconv4(c4)

        u3 = self.uconv3(d4)
        c3 = torch.cat([e2, u3], dim=1)
        d3 = self.dconv3(c3)

        u2 = self.uconv2(d3)
        c2 = torch.cat([e1, u2], dim=1)
        d2 = self.dconv2(c2)

        u1 = self.uconv1(d2)
        c1 = torch.cat([x, u1], dim=1)
        d1 = self.dconv1(c1)

        out = self.output(d1)
        return out


def build_model(model_name, cbam=True):

    if model_name == 'resunet10':
        model = BasicResUNet(1, 1, [1, 1, 1, 1], 64, cbam)
    elif model_name == 'resunet18':
        model = BasicResUNet(1, 1, [2, 2, 2, 2], 64, cbam)
    elif model_name == 'resunet34':
        model = BasicResUNet(1, 1, [3, 4, 6, 3], 64, cbam)
    elif model_name == 'resunet14':
        model = BottleneckResUNet(1, 1, [1, 1, 1, 1], 64, cbam)
    elif model_name == 'resunet26':
        model = BottleneckResUNet(1, 1, [2, 2, 2, 2], 64, cbam)
    elif model_name == 'resunet50':
        model = BottleneckResUNet(1, 1, [3, 4, 6, 3], 64, cbam)
    elif model_name == 'resunet101':
        model = BottleneckResUNet(1, 1, [3, 4, 23, 3], 64, cbam)
    else:
        raise ValueError('Invalid model_name')

    return model


if __name__ == '__main__':
    from torchsummary import summary

    resunet10 = build_model('resunet10')
    summary(resunet10, (1, 128, 128), batch_size=1)

    # resunet18 = build_model('resunet18')
    # summary(resunet18, (1, 128, 128), batch_size=1)

    # resunet34 = build_model('resunet34')
    # summary(resunet34, (1, 128, 128), batch_size=1)

    # resunet14 = build_model('resunet14')
    # summary(resunet14, (1, 128, 128), batch_size=1)

    # resunet26 = build_model('resunet26')
    # summary(resunet26, (1, 128, 128), batch_size=1)

    # resunet50 = build_model('resunet50')
    # summary(resunet50, (1, 128, 128), batch_size=1)

    # resunet101 = build_model('resunet101')
    # summary(resunet101, (1, 128, 128), batch_size=1)
