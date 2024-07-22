import torch
from torch import nn
from convs import Conv2DKAN
from kans import KAN

class KANBaseline(nn.Module):
    def __init__(self, input_channel, output_channel=None, n_classes=10, height=32, width=32, device='cpu'):
        super(KANBaseline, self).__init__()
        self.device = device
        self.conv_layers = nn.ModuleList([
            Conv2DKAN(input_channel, 2, 3, device=self.device),
            nn.AvgPool2d(3),
            Conv2DKAN(2, 2, 3, device=self.device),
        ]).to(self.device)
        self.kan_network = KAN([72, 36, n_classes], device=self.device, base_activation=nn.Tanh)
        # self.kan_network = KAN([128, 64, n_classes], device=self.device, base_activation=nn.Tanh)
    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.kan_network(x)
        return x