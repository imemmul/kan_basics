import torch
import numpy as np
from torch.nn import Module
from torch.nn.functional import tanh
from kanneuron import KANNeuron

class Layer(Module):
    def __init__(self, n_inputs, n_outputs, neuron_class=KANNeuron, device='cpu', **kwargs):
        super(Layer, self).__init__()
        self.n_inputs = n_inputs
        self.device = device
        self.neurons = [neuron_class(n_inputs, device=device, **kwargs) for _ in range(n_outputs)]
        self.xin = None
        self.xout = None
        self.dloss_dxin = None
        self.zero_grad()

    def forward(self, inputs):
        self.xin = inputs
        self.xout = torch.stack([neuron(self.xin) for neuron in self.neurons], dim=0).to(self.device)
        return self.xout

    def zero_grad(self, which=None):
        # reset gradients to zero
        if which is None:
            which = ['xin', 'weights', 'bias']
        for w in which:
            if w == 'xin':  # reset layer's d loss / d xin
                self.dloss_dxin = torch.zeros(self.n_inputs, device=self.device, dtype=torch.float32)
            elif w == 'weights':  # reset d loss / dw to zero for every neuron
                for nn in self.neurons:
                    nn.gradient_loss_wrt_weights = torch.zeros((self.n_inputs, nn.num_weights_per_edge), device=self.device, dtype=torch.float32)
            elif w == 'bias':  # reset d loss / db to zero for every neuron
                for nn in self.neurons:
                    nn.gradient_loss_wrt_bias = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            else:
                raise ValueError('input \'which\' value not recognized')

    def update_grad(self, ddelta):
        # Ensure ddelta is an array
        if torch.is_tensor(ddelta) and ddelta.ndim == 0:
            ddelta = torch.tensor([ddelta.item()] * len(self.neurons), device=self.device, dtype=torch.float32)
        for ii, ddelta_tmp in enumerate(ddelta):
            self.dloss_dxin += self.neurons[ii].derivative_output_wrt_input * ddelta_tmp
            self.neurons[ii].update_gradients(ddelta_tmp)
        return self.dloss_dxin


class KANNetwork(Module):
    def __init__(self, layers_shape, learning_rate, device='cpu', seed=None, **kwargs):
        super(KANNetwork, self).__init__()
        self.seed = torch.manual_seed(seed if seed else np.random.randint(int(1e4)))
        self.device = device
        self.layers_shape = layers_shape
        self.learning_rate = learning_rate
        self.n_layers = len(layers_shape) - 1
        self.layers = [Layer(layers_shape[i], layers_shape[i + 1], device=device, **kwargs) for i in range(self.n_layers)]

    def forward(self, x):
        x_in = x
        for li in range(self.n_layers):
            x_in = self.layers[li].forward(x_in)  # forward pass
        return x_in

    def backward(self, dloss_dy):
        delta = self.layers[-1].update_grad(dloss_dy)
        for ll in range(self.n_layers - 1)[::-1]:
            delta = self.layers[ll].update_grad(delta)
        return delta

    def update_weights(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.apply_gradient_descent(self.learning_rate)

    def zero_grad(self, which=None):
        if 'xin' in which:
            [layer.zero_grad(which=['xin']) for layer in self.layers]
        elif 'weights' in which and 'bias' in which:
            [layer.zero_grad(which=['weights', 'bias']) for layer in self.layers]