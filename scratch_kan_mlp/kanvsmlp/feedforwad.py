from kan import KANNeuron
from mlp import MLPNeuron
import numpy as np
from tqdm import tqdm
from loss import SquaredLoss, CrossEntropyLoss

class Layer:
    def __init__(self, n_inputs, n_outputs, neuron_class=KANNeuron, **kwargs):
        self.n_inputs = n_inputs
        self.neurons = [neuron_class(n_inputs) if (kwargs == {}) else neuron_class(n_inputs, **kwargs) for _ in range(n_outputs)]
        self.xin = None
        self.xout = None
        self.dloss_dxin = None
        self.zero_grad()
        
    def forward(self, inputs):
        self.xin = inputs
        self.xout = np.array([neuron(self.xin) for neuron in self.neurons])
        return self.xout

    def zero_grad(self, which=None):
        # reset gradients to zero
        if which is None:
            which = ['xin', 'weights', 'bias']
        for w in which:
            if w == 'xin':  # reset layer's d loss / d xin
                self.dloss_dxin = np.zeros(self.n_inputs)
            elif w == 'weights':  # reset d loss / dw to zero for every neuron
                for nn in self.neurons:
                    nn.gradient_loss_wrt_weights = np.zeros((self.n_inputs, self.neurons[0].num_weights_per_edge))
            elif w == 'bias':  # reset d loss / db to zero for every neuron
                for nn in self.neurons:
                    nn.gradient_loss_wrt_bias = 0
            else:
                raise ValueError('input \'which\' value not recognized')

    def update_grad(self, ddelta):
        # Ensure ddelta is an array
        if np.isscalar(ddelta):
            ddelta = np.array([ddelta] * len(self.neurons))
        for ii, ddelta_tmp in enumerate(ddelta):
            self.dloss_dxin += self.neurons[ii].derivative_output_wrt_input * ddelta_tmp
            self.neurons[ii].update_gradients(ddelta_tmp)
        return self.dloss_dxin

class KANNetwork:
    def __init__(self, layers_shape, learning_rate, seed=None, loss=SquaredLoss, **kwargs):
        self.seed = np.random.randint(int(1e4)) if seed is None else int(seed)
        np.random.seed(self.seed)
        self.layers_shape = layers_shape
        self.learning_rate = learning_rate
        self.n_layers = len(layers_shape) - 1
        self.layers = [Layer(layers_shape[i], layers_shape[i+1], **kwargs) for i in range(self.n_layers)]
        
    def forward(self, x):
        x_in = x
        for li in range(self.n_layers):
            x_in = self.layers[li].forward(x_in) # forward pass
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