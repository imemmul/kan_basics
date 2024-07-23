import torch
from torch import nn
from convs import Conv2DKAN, Conv2DFastKAN
from kans import KAN, FastKAN
import torch.nn

class KANBaseline(nn.Module):
    def __init__(self, input_channel, output_channel=None, n_classes=10, height=32, width=32, device='cpu', fastkan=False):
        super(KANBaseline, self).__init__()
        self.device = device
        if not fastkan:
            self.conv_layers = nn.ModuleList([
                Conv2DKAN(input_channel, 2, 3, device=self.device),
                nn.AvgPool2d(3),
                Conv2DKAN(2, 2, 3, device=self.device),
            ]).to(self.device)
            # self.kan_network = KAN([72, 36, n_classes], device=self.device, base_activation=nn.SiLU)
            self.kan_network = KAN([128, 64, n_classes], device=self.device, base_activation=nn.SiLU)
            # self.kan_network = KAN([32, 16, n_classes], device=self.device, base_activation=nn.Tanh)
        else:
            self.conv_layers = nn.ModuleList([
                Conv2DFastKAN(input_channel, 2, 3, device=self.device),
                nn.AvgPool2d(3),
                Conv2DFastKAN(2, 2, 3, device=self.device),
            ]).to(self.device)
            # self.kan_network = FastKAN([72, 36, n_classes], base_activation=nn.functional.silu).to(self.device)
            self.kan_network = FastKAN([128, 64, n_classes], base_activation=nn.functional.silu).to(self.device)
    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.kan_network(x)
        return x