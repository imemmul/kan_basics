from basic_neuron import Neuron
from act import relu, silu, tanh_act, sigmoid
import numpy as np

class MLPNeuron(Neuron):
    def __init__(self, num_inputs, weights_range=None, activation=relu):
        super().__init__(num_inputs, num_weights_per_edge=1, weights_range=weights_range)
        self.activation = activation
        self.activation_input = None

    def compute_edge_values(self):
        self.edge_values = self.weights[:, 0] * self.input_values

    def compute_output_value(self):
        self.activation_input = sum(self.edge_values.flatten()) + self.bias # is flatten will fix this issue? 
        self.output_value = self.activation(self.activation_input, get_derivative=False)

    def compute_derivative_output_wrt_edge(self):
        self.derivative_output_wrt_edge = self.activation(self.activation_input, get_derivative=True) * np.ones(self.num_inputs)

    def compute_derivative_output_wrt_bias(self):
        self.derivative_output_wrt_bias = self.activation(self.activation_input, get_derivative=True)

    def compute_derivative_edge_wrt_weights(self):
        self.derivative_edge_wrt_weights = np.reshape(self.input_values, (-1, 1))

    def compute_derivative_edge_wrt_input(self):
        self.derivative_edge_wrt_input = self.weights.flatten()