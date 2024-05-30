import torch
from torch import nn
from torchvision.models import resnet18

class ResNet(nn.Module):
    def __init__(self, num_features=256):
        super(ResNet, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_features)

    def forward(self, x):
        return self.resnet(x)
