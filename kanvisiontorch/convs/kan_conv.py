from kans import KAN
import torch.nn as nn
import torch
import torch.nn.functional as F

class KANClassification(nn.Module):
    def __init__(self, input_channel, output_channel=None, n_classes=10, height=32, width=32, device='cpu'):
        super(KANClassification, self).__init__()
        self.device = device
        self.conv_layers = nn.ModuleList([
            ConvKANLayer(input_channel, 4, 3, device=self.device),
            nn.MaxPool2d(3),
            ConvKANLayer(4, 2, 3, device=self.device),
        ]).to(self.device)

        self.kan_network = KAN([72, 36, n_classes], device=self.device)

    def forward(self, x):
        for layer in self.conv_layers:
            # print(f"before layer: {x.shape}")
            x = layer(x)
            # print(f"after layer: {x.shape}")
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        x = x.view(x.size(0), -1)
        # print(x.shape)
        print(f"x_shape: {x.shape}")
        x = self.kan_network(x)
        return x

class ConvKANLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device='cpu'):
        super(ConvKANLayer, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kan_layers = nn.ModuleList()

        for i in range(out_channels):
            self.kan_network = KAN([in_channels * kernel_size * kernel_size, 1], device=self.device)
            self.kan_layers.append(self.kan_network)

    def forward(self, x):
        x = x.to(self.device)
        batch_size, in_channels, in_height, in_width = x.shape
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1

        out = torch.zeros((batch_size, self.out_channels, out_height, out_width), dtype=torch.float32, device=self.device)
        self.xin_cache = x

        for b in range(batch_size):
            for o in range(self.out_channels):
                for i in range(0, in_height - self.kernel_size + 1, self.stride):
                    for j in range(0, in_width - self.kernel_size + 1, self.stride):
                        region = x[b, :, i:i + self.kernel_size, j:j + self.kernel_size]
                        out[b, o, i // self.stride, j // self.stride] = self.kan_layers[o](region.flatten().unsqueeze(0))

        self.output = out
        return out