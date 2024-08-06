import torch
from torch import nn

class NNBaseline(nn.Module):
    def __init__(self, input_channel, hidden_channels=10, n_classes=10, height=32, width=32, device='cpu', **kwargs):
        super(NNBaseline, self).__init__()
        self.device = device
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ).to(self.device)
        
        # Calculate the size of the flattened feature map after convolution and pooling
        self.feature_map_size = (height // 2) * (width // 2) * hidden_channels
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_map_size, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes),
        ).to(self.device)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x