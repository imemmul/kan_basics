import torch.nn as nn
from convs import Conv2DFastKAN, Conv2DKAN
import torch
from kans import FastKAN, KAN
from models import KANBaseline

class TinyVGGKAN(KANBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_cls = Conv2DKAN
        self.fc_cls = KAN
        if self.use_fastkan:
            self.conv_cls = Conv2DFastKAN
            self.fc_cls = FastKAN
        self.conv_layers = nn.ModuleList([
            self.conv_cls(in_channels=self.input_channel, out_channels=self.hidden_channels, kernel_size=3, stride=1, device=self.device),
            self.conv_cls(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, device=self.device),
            nn.MaxPool2d(2),
            self.conv_cls(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, device=self.device),
            self.conv_cls(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, device=self.device),
            nn.MaxPool2d(2)
            
        ])
        dummy_input = torch.randn(1, self.input_channel, self.height, self.width).to(self.device)
        out = dummy_input
        for layer in self.conv_layers:
            out = layer(out)
        conv_output_size = out.view(out.size(0), -1).size(1)
        self.fc = self.fc_cls([conv_output_size, conv_output_size//2, self.n_classes]).to(self.device)
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x