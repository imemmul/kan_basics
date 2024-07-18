import torch
import torch.nn as nn
import torch.nn.functional as F
from kannetworktorch import KANNetwork
from kanneuron import KANNeuron
from getsplines import get_bsplines_torch



class KANClassification(nn.Module):
    def __init__(self, input_channel, output_channel=None, n_classes=10, height=32, width=32, learning_rate=0.01, device='cpu'):
        super(KANClassification, self).__init__()
        self.device = device
        self.conv_layers = nn.ModuleList([
            ConvKANLayer(input_channel, 2, 3, device=device),
            ConvKANLayer(2, 1, 3, device=device),
        ]).to(self.device)

        self.kan_network = KANNetwork(
            [1 * (height - 4) * (width - 4), 32, n_classes],  # Adjusted layer size
            neuron_class=KANNeuron,
            x_bounds=[-1, 1],  # input domain bounds
            get_edge_fun=get_bsplines_torch,  # edge function type (B-splines or Chebyshev)
            seed=472,
            learning_rate=learning_rate,
            n_weights_per_edge=7,  # n. edge functions
            weights_range=[-1, 1],
            device=self.device
        ).to(self.device)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer.forward(x)
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(x.shape)
        x = x.flatten()
        x = self.kan_network.forward(x)
        return x

    def backward(self, dloss_dy):
        delta = self.kan_network.backward(dloss_dy)
        delta = delta.view(-1, self.conv_layers[-1].out_channels,
                           (self.conv_layers[-1].xin_cache.size(2) - self.conv_layers[-1].kernel_size) // self.conv_layers[-1].stride + 1,
                           (self.conv_layers[-1].xin_cache.size(3) - self.conv_layers[-1].kernel_size) // self.conv_layers[-1].stride + 1)
        for layer in self.conv_layers[::-1]:
            delta = layer.backward(delta)

    def update(self):
        self.kan_network.update_weights()
        for layer in self.conv_layers:
            for kan_layer in layer.kan_layers:
                kan_layer.update_weights()

    def zero_grad(self, which=None):
        self.kan_network.zero_grad(which)
        for layer in self.conv_layers:
            for kan_layer in layer.kan_layers:
                kan_layer.zero_grad(which)


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
            self.kan_network = KANNetwork(
                [in_channels * kernel_size * kernel_size, 1],  # layer size
                neuron_class=KANNeuron,
                x_bounds=[-1, 1],  # input domain bounds
                get_edge_fun=get_bsplines_torch,  # edge function type (B-splines or Chebyshev)
                seed=472,
                learning_rate=0.01,
                n_weights_per_edge=7,  # n. edge functions
                weights_range=[-1, 1],
                device=self.device
            ).to(self.device)
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
                        out[b, o, i // self.stride, j // self.stride] = self.kan_layers[o].forward(region.flatten())

        self.output = out
        return out

    def backward(self, dloss_dy):
        batch_size, out_channels, out_height, out_width = dloss_dy.shape
        dloss_dx = torch.zeros_like(self.xin_cache, dtype=torch.float32, device=self.device)

        for b in range(batch_size):
            for o in range(out_channels):
                for i in range(0, out_height * self.stride, self.stride):
                    for j in range(0, out_width * self.stride, self.stride):
                        region = self.xin_cache[b, :, i:i + self.kernel_size, j:j + self.kernel_size]
                        dloss_dout = dloss_dy[b, o, i // self.stride, j // self.stride]
                        dloss_dregion = self.kan_layers[o].backward(torch.tensor([dloss_dout], dtype=torch.float32, device=self.device))
                        dloss_dx[b, :, i:i + self.kernel_size, j:j + self.kernel_size] += dloss_dregion.view(self.in_channels, self.kernel_size, self.kernel_size)

        return dloss_dx