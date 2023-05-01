import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

class Down(nn.Module):
    def __init__(self, num_channels, in_channels, out_channels):
        super(Down, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]*(num_channels - 1)
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class Up(nn.Module):
    def __init__(self, num_channels, input_channels, output_channels, last=False):
        super(Up, self).__init__()
        layers = [
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        ]*(num_channels - 1)
        layers += [
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        ]
        if not last:
            layers.append(nn.BatchNorm2d(output_channels)),
            layers.append(nn.ReLU(inplace=True)),
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)

class SegNet(nn.Module):
    def __init__(self, input_channels, num_classes, weights=None):
        super(SegNet, self).__init__()

        # vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', weights=weights)
        # VGG16_BN_Weights.IMAGENET1K_V1
        # features = list(vgg.features.children())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)

        self.down1 = Down(2, input_channels, 64)
        self.down2 = Down(2, 64, 128)
        self.down3 = Down(3, 128, 256)
        self.down4 = Down(3, 256, 512)
        self.down5 = Down(3, 512, 512)

        # self.down1 = nn.Sequential(*features[:6])
        # self.down2 = nn.Sequential(*features[7:13])
        # self.down3 = nn.Sequential(*features[14:23])
        # self.down4 = nn.Sequential(*features[24:33])
        # self.down5 = nn.Sequential(*features[34:43])

        self.up5 = Up(3, 512, 512)
        self.up4 = Up(3, 512, 256)
        self.up3 = Up(3, 256, 128)
        self.up2 = Up(2, 128, 64)
        self.up1 = Up(2, 64, num_classes, last=True)

    def forward(self, x):
        # Encode phase
        x = self.down1(x)
        x1_size = x.size()
        x, indices_1 = self.pool(x)

        x = self.down2(x)
        x2_size = x.size()
        x, indices_2 = self.pool(x)

        x = self.down3(x)
        x3_size = x.size()
        x, indices_3 = self.pool(x)

        x = self.down4(x)
        x4_size = x.size()
        x, indices_4 = self.pool(x)

        x = self.down5(x)
        x5_size = x.size()
        x, indices_5 = self.pool(x)

        # Decode phase
        x = self.unpool(x, indices_5, output_size=x5_size)
        x = self.up5(x)

        x = self.unpool(x, indices_4, output_size=x4_size)
        x = self.up4(x)

        x = self.unpool(x, indices_3, output_size=x3_size)
        x = self.up3(x)

        x = self.unpool(x, indices_2, output_size=x2_size)
        x = self.up2(x)

        x = self.unpool(x, indices_1, output_size=x1_size)
        x = self.up1(x)

        x = self.softmax(x)
        return x
