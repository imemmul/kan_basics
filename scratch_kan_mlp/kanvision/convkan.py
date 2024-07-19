import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from kanvsmlp import KANNetwork, KANNeuron, CrossEntropyLoss, get_bsplines
import numpy as np

class KANClassification:
    def __init__(self, input_channel, output_channel=None, n_classes=10, height=32, width=32, learning_rate=0.01):
        self.conv_layers = [
            ConvKANLayer(input_channel, 4, 5),
            ConvKANLayer(4, 2, 5),
            # ConvKANLayer(4, 2, 5)
        ]
        self.kan_network = KANNetwork(
            [2 * (height - 8) * (width - 8), 64, n_classes],  # Adjusted layer size
            neuron_class=KANNeuron, 
            x_bounds=[-1, 1],  # input domain bounds
            get_edge_fun=get_bsplines,  # edge function type (B-splines ot Chebyshev)
            seed=472,
            learning_rate=learning_rate,
            n_weights_per_edge=7,  # n. edge functions
            weights_range=[-1, 1]
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer.forward(x)
        x = x.flatten()
        x = self.kan_network.forward(x)
        return x

    def backward(self, dloss_dy):
        delta = self.kan_network.backward(dloss_dy)
        delta = delta.reshape((-1, self.conv_layers[-1].out_channels, 
                               (self.conv_layers[-1].xin_cache.shape[2] - self.conv_layers[-1].kernel_size) // self.conv_layers[-1].stride + 1,
                               (self.conv_layers[-1].xin_cache.shape[3] - self.conv_layers[-1].kernel_size) // self.conv_layers[-1].stride + 1))
        for layer in self.conv_layers[::-1]:
            delta = layer.backward(delta)

    def update(self):
        self.kan_network.update_weights(self.kan_network.learning_rate)
        for layer in self.conv_layers:
            for kan_layer in layer.kan_layers:
                kan_layer.update_weights(self.kan_network.learning_rate)

    def zero_grad(self, which=None):
        self.kan_network.zero_grad(which)
        for layer in self.conv_layers:
            for kan_layer in layer.kan_layers:
                kan_layer.zero_grad(which)

class ConvKANLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kan_layers = []
        for i in range(out_channels):
            self.kan_network = KANNetwork(
                [in_channels * kernel_size * kernel_size, 1],  # layer size
                neuron_class=KANNeuron, 
                x_bounds=[-1, 1],  # input domain bounds
                get_edge_fun=get_bsplines,  # edge function type (B-splines ot Chebyshev)
                seed=472,
                learning_rate=0.01,
                n_weights_per_edge=7,  # n. edge functions
                weights_range=[-1, 1]
            )
            self.kan_layers.append(self.kan_network)
                                        
    def forward(self, x):
        x = x.astype(np.float32)  # Ensure input is of type float32
        batch_size, in_channels, in_height, in_width = x.shape

        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1

        out = np.zeros((batch_size, self.out_channels, out_height, out_width), dtype=np.float32)
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
        dloss_dx = np.zeros_like(self.xin_cache, dtype=np.float32)

        for b in range(batch_size):
            for o in range(out_channels):
                for i in range(0, out_height * self.stride, self.stride):
                    for j in range(0, out_width * self.stride, self.stride):
                        region = self.xin_cache[b, :, i:i + self.kernel_size, j:j + self.kernel_size]
                        dloss_dout = dloss_dy[b, o, i // self.stride, j // self.stride]
                        dloss_dregion = self.kan_layers[o].backward(np.array([dloss_dout], dtype=np.float32))
                        dloss_dx[b, :, i:i + self.kernel_size, j:j + self.kernel_size] += dloss_dregion.reshape(self.in_channels, self.kernel_size, self.kernel_size)

        return dloss_dx

    def backward(self, dloss_dy):
        batch_size, out_channels, out_height, out_width = dloss_dy.shape
        dloss_dx = np.zeros_like(self.xin_cache)

        for b in range(batch_size):
            for o in range(out_channels):
                for i in range(0, out_height * self.stride, self.stride):
                    for j in range(0, out_width * self.stride, self.stride):
                        region = self.xin_cache[b, :, i:i + self.kernel_size, j:j + self.kernel_size]
                        dloss_dout = dloss_dy[b, o, i // self.stride, j // self.stride]
                        dloss_dregion = self.kan_layers[o].backward(np.array([dloss_dout]))
                        dloss_dx[b, :, i:i + self.kernel_size, j:j + self.kernel_size] += dloss_dregion.reshape(self.in_channels, self.kernel_size, self.kernel_size)

        return dloss_dx

if __name__ == "__main__":
    x = np.random.randn(8, 1, 32, 32)
    layers = [
        ConvKANLayer(1, 8, 5),
        ConvKANLayer(8, 4, 5),
        ConvKANLayer(4, 2, 5)
    ]
    
    for layer in layers:
        x = layer.forward(x)
        print(x.shape)
    
    y = np.random.randint(0, 10, (8,), )
    # for i in range(8):
    #     y_onehot = np.zeros((10,))
    #     y_onehot[y[i]] = 1
    #     y[i] = y_onehot
    #     x[i] = x[i] / np.max(x[i])
    #     model = KANClassification(1, 10)
