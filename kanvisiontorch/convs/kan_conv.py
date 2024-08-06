from kans import KAN, FastKAN
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import imageio
import numpy as np

# TODO so BatchNorm needed to grad update spline_weights, i guess they are exploded while feedforwarded.

# CONVOLUTIONAL KOLMOGOROV ARNOLD NETWORKS
class Conv2DKAN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device='cpu'):
        super(Conv2DKAN, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kan_layers = nn.ModuleList()
        self.attention_maps = []
        for i in range(out_channels):
            kan_networks = KAN([in_channels * kernel_size * kernel_size, in_channels * kernel_size * kernel_size], device=self.device, base_activation=nn.SiLU)
            self.kan_layers.append(kan_networks)
        self.bn = nn.BatchNorm2d(out_channels).to(self.device)
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
        self.attention_maps.append(out.detach().cpu().numpy())
        return self.bn(out)

    def save_attention_gif(self, path='attention_maps.gif'):
        frames = []
        for maps in self.attention_maps:
            for i in range(maps.shape[1]):  # Loop over channels
                frame = maps[0, i]  # Assuming batch_size is 1 for visualization
                frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize to [0, 1]
                frame = (frame * 255).astype(np.uint8)  # Convert to [0, 255]
                frames.append(frame)

        imageio.mimsave(path, frames, fps=2)  # Save frames as a GIF
    
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
        for b in range(batch_size):
            for o in range(self.out_channels):
                for i in range(0, in_height - self.kernel_size + 1, self.stride):
                    for j in range(0, in_width - self.kernel_size + 1, self.stride):
                        region = x[b, :, i:i + self.kernel_size, j:j + self.kernel_size]
                        out[b, o, i // self.stride, j // self.stride] = torch.sum(self.kan_layers[o](region.flatten().unsqueeze(0)))
        self.output = out
        return out


class Conv2DFastKAN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device='cpu', use_base_update=False, use_layernorm=False):
        super(Conv2DFastKAN, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kan_layers = nn.ModuleList()
        self.attention_maps = []
        for i in range(out_channels):
            kan_networks = FastKAN([in_channels * kernel_size * kernel_size, in_channels * kernel_size * kernel_size], use_base_update=use_base_update, use_layernorm=use_layernorm).to(self.device)
            self.kan_layers.append(kan_networks)
        self.bn = nn.BatchNorm2d(out_channels).to(self.device)
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
        self.attention_maps.append(out.detach().cpu().numpy())
        return self.bn(out)

    def save_attention_gif(self, path):
        frames = []
        for attention_map in self.attention_maps:
            for channel in range(attention_map.shape[1]):
                fig, ax = plt.subplots()
                cax = ax.matshow(attention_map[0, channel, :, :], cmap='viridis')
                fig.colorbar(cax)
                fig.canvas.draw()
                
                # Convert canvas to image
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)
                plt.close(fig)
        
        if frames:
            imageio.mimsave(path, frames, fps=2)
        else:
            raise ValueError("No frames to save")