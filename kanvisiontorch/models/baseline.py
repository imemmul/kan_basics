import torch
from torch import nn
from convs import Conv2DKAN, Conv2DFastKAN
from kans import KAN, FastKAN
import torch.nn
from torch.profiler import profile, record_function

class KANBaseline(nn.Module):
    def __init__(self, input_channel, hidden_channels=10, n_classes=10, height=32, width=32, device='cpu', use_fastkan=False):
        super(KANBaseline, self).__init__()
        self.device = device
        self.use_fastkan = use_fastkan
        self.input_channel = input_channel
        self.hidden_channels = hidden_channels
        self.n_classes = n_classes
        self.height = height
        self.width = width
        self.device = device
        if self.use_fastkan:
            self.conv_layers = nn.ModuleList([
                Conv2DFastKAN(input_channel, hidden_channels, 3, device=self.device, use_base_update=False, use_layernorm=False),
                nn.MaxPool2d(2),
            ]).to(self.device)
            # self.kan_network = KAN([72, 36, n_classes], device=self.device, base_activation=nn.SiLU)
            # self.kan_network = FastKAN([32*32*3, 16*16*3, 64*3, n_classes], base_activation=nn.functional.silu).to(self.device)
            self.kan_network = FastKAN([5408, 100, n_classes], base_activation=nn.functional.silu, use_base_update=True, use_layernorm=True).to(self.device)
        else:
            self.conv_layers = nn.ModuleList([
                Conv2DKAN(input_channel, hidden_channels, 3, device=self.device),
                nn.MaxPool2d(2),
            ]).to(self.device)
            self.kan_network = KAN([5408, 100, n_classes], device=self.device, base_activation=nn.SiLU)
            # self.kan_network = KAN([128, 64, n_classes], device=self.device, base_activation=nn.SiLU)
            # self.kan_network = KAN([32, 16, n_classes], device=self.device, base_activation=nn.Tanh)
    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            with record_function(f"ConvKANLayer_{i}"):
                x = layer(x)
        
        x = x.view(x.size(0), -1)
        with record_function("KANLinear"):
            x = self.kan_network(x)
        return x