import torch
from torchvision import models
from torchsummary import summary
from architecture import SegNet

# vgg16 = models.vgg16_bn()
# print(summary(vgg16, (3, 128, 128), device='cpu'))
# print(list(vgg16.features.children()))
# vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', weights=None)
# features = list(vgg.features.children())
# print(features)
# print(len(features))

seg = SegNet(3, 10)
print(summary(seg, (3, 100, 128), device='cpu'))
