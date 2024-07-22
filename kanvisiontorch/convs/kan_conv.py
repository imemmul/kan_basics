from kans import KAN, KANLayer
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class KANClassification(nn.Module):
    def __init__(self, input_channel, output_channel=None, n_classes=10, height=32, width=32, device='cpu'):
        super(KANClassification, self).__init__()
        self.device = device
        self.conv_layers = nn.ModuleList([
            ConvKANLayer(input_channel, 2, 3, device=self.device),
            nn.AvgPool2d(3),
            ConvKANLayer(2, 2, 3, device=self.device),
            nn.BatchNorm2d(2),
        ]).to(self.device)

        self.kan_network = KAN([128, 64, n_classes], device=self.device)

    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            # print(f"Conv layer {i}: ")
            x = layer(x)
        x = x.view(x.size(0), -1)
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
            kan_networks = KAN([in_channels * kernel_size * kernel_size, in_channels * kernel_size * kernel_size], device=self.device)
            self.kan_layers.append(kan_networks)
            
    def forward(self, x):
        x = x.to(self.device)
        batch_size, in_channels, in_height, in_width = x.shape

        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        x_unf = x_unf.view(batch_size, in_channels, self.kernel_size, self.kernel_size, -1)
        x_unf = x_unf.permute(0, 4, 1, 2, 3).contiguous()
        x_unf = x_unf.view(batch_size * x_unf.size(1), -1)
        out = torch.zeros(batch_size, self.out_channels, out_height * out_width, dtype=torch.float32, device=self.device)
        for o in range(self.out_channels):
            kan_out = self.kan_layers[o](x_unf).view(batch_size, out_height * out_width, -1)
            kan_out = torch.sum(kan_out, dim=-1)
            out[:, o, :] = kan_out
        out = out.view(batch_size, self.out_channels, out_height, out_width)
        return out
    
    def forward_deprecated(self, x):
        """
        shitty conv below, behold !
        """
        x = x.to(self.device)
        batch_size, in_channels, in_height, in_width = x.shape
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1

        out = torch.zeros((batch_size, self.out_channels, out_height, out_width), dtype=torch.float32, device=self.device)
        self.xin_cache = x
        # total_time_kan_forward = 0
        # total_time_conv_kan = time.time()
        print(f"how many loops are there below: {batch_size * self.out_channels * (in_height - self.kernel_size + 1) * (in_width - self.kernel_size + 1)}")
        for b in range(batch_size):
            for o in range(self.out_channels):
                for i in range(0, in_height - self.kernel_size + 1, self.stride):
                    for j in range(0, in_width - self.kernel_size + 1, self.stride):
                        region = x[b, :, i:i + self.kernel_size, j:j + self.kernel_size]
                        # conv_forward = time.time()
                        # print(f"region shape: {region.shape}, kan shape: {region.flatten().unsqueeze(0).shape}")
                        # print(f"output_kan shape: {output_kan.shape}, torch.sum(output_kan): {torch.sum(output_kan).shape}")
                        
                        out[b, o, i // self.stride, j // self.stride] = torch.sum(self.kan_layers[o](region.flatten().unsqueeze(0)))
                        # conv_forward_time = time.time() - conv_forward
                        # total_time_kan_forward += conv_forward_time
        
        # total_time_conv_kan = time.time() - total_time_conv_kan
        # print(f"All time: {total_time_conv_kan}, total time taken for kan forward: {total_time_kan_forward}")
        self.output = out
        return out