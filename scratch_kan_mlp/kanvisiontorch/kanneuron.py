import torch
from activations import tanh_act
from getsplines import get_bsplines, get_bsplines_torch

import torch
from torch import tensor
from torch.nn.functional import silu
from scipy.interpolate import BSpline
import numpy as np

class Neuron:
    def __init__(self, num_inputs, num_weights_per_edge, weights_range=None, device='cpu'):
        self.num_inputs = num_inputs
        self.num_weights_per_edge = num_weights_per_edge
        weights_range = [-1, 1] if weights_range is None else weights_range
        self.weights = torch.tensor(np.random.uniform(weights_range[0], weights_range[-1], size=(self.num_inputs, self.num_weights_per_edge)), device=device, dtype=torch.float32)
        self.bias = 0 # No bias in KAN
        self.device = device
        self.input_values = None
        self.edge_values = None
        self.output_value = None

        self.derivative_output_wrt_edge = None
        self.derivative_output_wrt_bias = None
        self.derivative_edge_wrt_weights = None
        self.derivative_edge_wrt_input = None
        self.derivative_output_wrt_input = None
        self.derivative_output_wrt_weights = None

        self.gradient_loss_wrt_weights = torch.zeros((self.num_inputs, self.num_weights_per_edge), device=device, dtype=torch.float32)
        self.gradient_loss_wrt_bias = 0

    def __call__(self, input_values):
        self.input_values = torch.tensor(input_values, device=self.device, dtype=torch.float32)
        self.compute_edge_values()
        self.compute_output_value()

        self.compute_derivative_output_wrt_edge()
        self.compute_derivative_output_wrt_bias()
        self.compute_derivative_edge_wrt_weights()
        self.compute_derivative_edge_wrt_input()
        assert self.derivative_output_wrt_edge.shape == (self.num_inputs,)
        assert self.derivative_edge_wrt_input.shape == (self.num_inputs,)
        assert self.derivative_edge_wrt_weights.shape == (self.num_inputs, self.num_weights_per_edge)
        
        self.compute_derivative_output_wrt_input()
        self.compute_derivative_output_wrt_weights()

        return self.output_value

    def compute_edge_values(self):
        pass

    def compute_output_value(self):
        pass

    def compute_derivative_output_wrt_edge(self):
        pass

    def compute_derivative_output_wrt_bias(self):
        pass

    def compute_derivative_edge_wrt_weights(self):
        pass

    def compute_derivative_edge_wrt_input(self):
        pass

    def compute_derivative_output_wrt_input(self):
        self.derivative_output_wrt_input = self.derivative_output_wrt_edge * self.derivative_edge_wrt_input

    def compute_derivative_output_wrt_weights(self):
        self.derivative_output_wrt_weights = torch.diag(self.derivative_output_wrt_edge) @ self.derivative_edge_wrt_weights

    def update_gradients(self, derivative_loss_wrt_output):
        self.gradient_loss_wrt_weights += self.derivative_output_wrt_weights * derivative_loss_wrt_output
        self.gradient_loss_wrt_bias += self.derivative_output_wrt_bias * derivative_loss_wrt_output

    def zero_grad(self):
        self.gradient_loss_wrt_weights.fill_(0)
        self.gradient_loss_wrt_bias.fill_(0)

    def apply_gradient_descent(self, learning_rate):
        self.weights -= learning_rate * self.gradient_loss_wrt_weights
        self.bias -= learning_rate * self.gradient_loss_wrt_bias

class KANNeuron(Neuron):

    def __init__(self, n_in, n_weights_per_edge, x_bounds, weights_range=None, get_edge_fun=get_bsplines_torch, device='cpu', **kwargs):
        self.x_bounds = x_bounds
        super().__init__(n_in, num_weights_per_edge=n_weights_per_edge, weights_range=weights_range, device=device)
        self.edge_fun, self.edge_fun_der = get_edge_fun(self.x_bounds, self.num_weights_per_edge, **kwargs)
        self.device = device

    def compute_edge_values(self):
        # apply edge functions
        self.phi_x_mat = torch.stack([self.edge_fun[b](self.input_values) for b in self.edge_fun], dim=1).to(self.device)
        self.phi_x_mat[torch.isnan(self.phi_x_mat)] = 0
        self.edge_values = torch.sum(self.weights * self.phi_x_mat.to(self.device), dim=1)  # Ensure weights are also on the correct device

    def compute_output_value(self):
        # note: node function <- tanh to avoid any update of spline grids
        self.output_value = tanh_act(torch.sum(self.edge_values), get_derivative=False)

    def compute_derivative_output_wrt_edge(self):
        self.derivative_output_wrt_edge = tanh_act(torch.sum(self.edge_values), get_derivative=True) * torch.ones(self.num_inputs, device=self.device)

    def compute_derivative_edge_wrt_weights(self):
        self.derivative_edge_wrt_weights = self.phi_x_mat

    def compute_derivative_edge_wrt_input(self):
        phi_x_der_mat = torch.stack([self.edge_fun_der[b](self.input_values) if self.edge_fun[b](self.input_values) is not None else torch.zeros_like(self.input_values, device=self.device) for b in self.edge_fun_der], dim=1)
        phi_x_der_mat[torch.isnan(phi_x_der_mat)] = 0
        self.derivative_edge_wrt_input = torch.sum(self.weights * phi_x_der_mat, dim=1)

    def compute_derivative_output_wrt_bias(self):
        # no bias in KAN!
        self.derivative_output_wrt_bias = 0


# class Neuron:
#     def __init__(self, num_inputs, num_weights_per_edge, weights_range=None, device="cpu"):
#         self.num_inputs = num_inputs
#         self.num_weights_per_edge = num_weights_per_edge
#         self.device = device
#         weights_range = [-1, 1] if weights_range is None else weights_range
#         self.weights = torch.FloatTensor(num_inputs, num_weights_per_edge).uniform_(weights_range[0], weights_range[-1]).to(self.device)
#         self.bias = torch.FloatTensor(1).uniform_(weights_range[0], weights_range[-1]).to(self.device)
#         self.input_values = None
#         self.edge_values = None
#         self.output_value = None

#         self.derivative_output_wrt_edge = None
#         self.derivative_output_wrt_bias = None
#         self.derivative_edge_wrt_weights = None
#         self.derivative_edge_wrt_input = None
#         self.derivative_output_wrt_input = None
#         self.derivative_output_wrt_weights = None

#         self.gradient_loss_wrt_weights = torch.zeros(num_inputs, num_weights_per_edge, device=self.device)
#         self.gradient_loss_wrt_bias = torch.zeros(1, device=self.device)

#     def __call__(self, input_values):
#         self.input_values = torch.tensor(input_values, device=self.device, dtype=torch.float32)
#         self.compute_edge_values()
#         self.compute_output_value()

#         self.compute_derivative_output_wrt_edge()
#         self.compute_derivative_output_wrt_bias()
#         self.compute_derivative_edge_wrt_weights()
#         self.compute_derivative_edge_wrt_input()
#         assert self.derivative_output_wrt_edge.shape == (self.num_inputs,)
#         assert self.derivative_edge_wrt_input.shape == (self.num_inputs,)
#         assert self.derivative_edge_wrt_weights.shape == (self.num_inputs, self.num_weights_per_edge)
        
#         self.compute_derivative_output_wrt_input()
#         self.compute_derivative_output_wrt_weights()

#         return self.output_value

#     def compute_edge_values(self):
#         pass

#     def compute_output_value(self):
#         pass

#     def compute_derivative_output_wrt_edge(self):
#         pass

#     def compute_derivative_output_wrt_bias(self):
#         pass

#     def compute_derivative_edge_wrt_weights(self):
#         pass

#     def compute_derivative_edge_wrt_input(self):
#         pass

#     def compute_derivative_output_wrt_input(self):
#         self.derivative_output_wrt_input = self.derivative_output_wrt_edge * self.derivative_edge_wrt_input

#     def compute_derivative_output_wrt_weights(self):
#         self.derivative_output_wrt_weights = torch.diag(self.derivative_output_wrt_edge) @ self.derivative_edge_wrt_weights

#     def update_gradients(self, derivative_loss_wrt_output):
#         self.gradient_loss_wrt_weights += self.derivative_output_wrt_weights * derivative_loss_wrt_output
#         self.gradient_loss_wrt_bias += self.derivative_output_wrt_bias * derivative_loss_wrt_output

#     def zero_grad(self):
#         self.gradient_loss_wrt_weights.zero_()
#         self.gradient_loss_wrt_bias.zero_()

#     def apply_gradient_descent(self, learning_rate):
#         self.weights -= learning_rate * self.gradient_loss_wrt_weights
#         self.bias -= learning_rate * self.gradient_loss_wrt_bias


# class KANNeuron(Neuron):
#     def __init__(self, n_in, n_weights_per_edge, x_bounds, weights_range=None, get_edge_fun=get_bsplines_torch, device="cpu", **kwargs):
#         self.x_bounds = x_bounds
#         self.device = device
#         super().__init__(n_in, num_weights_per_edge=n_weights_per_edge, weights_range=weights_range)
#         self.edge_fun, self.edge_fun_der = get_edge_fun(self.x_bounds, self.num_weights_per_edge, device=device, **kwargs)

#     def compute_edge_values(self):
#         # apply edge functions
#         print(self.edge_fun)
#         self.phi_x_mat = torch.stack([self.edge_fun[b](self.input_values) for b in self.edge_fun], dim=1).to(self.device)
#         self.phi_x_mat[torch.isnan(self.phi_x_mat)] = 0
#         self.edge_values = torch.sum(self.weights * self.phi_x_mat, dim=1)

#     def compute_output_value(self):
#         # note: node function <- tanh to avoid any update of spline grids
#         self.output_value = tanh_act(torch.sum(self.edge_values.flatten()), get_derivative=False)

#     def compute_derivative_output_wrt_edge(self):
#         self.derivative_output_wrt_edge = tanh_act(torch.sum(self.edge_values.flatten()), get_derivative=True) * torch.ones(self.num_inputs, device=self.device)

#     def compute_derivative_edge_wrt_weights(self):
#         self.derivative_edge_wrt_weights = self.phi_x_mat

#     def compute_derivative_edge_wrt_input(self):
#         phi_x_der_mat = torch.stack([self.edge_fun_der[b](self.input_values) if self.edge_fun[b](self.input_values) is not None else torch.tensor(0, device=self.device)
#                                      for b in self.edge_fun_der], dim=1)  # shape (n_in, n_weights_per_edge)
#         phi_x_der_mat[torch.isnan(phi_x_der_mat)] = 0
#         self.derivative_edge_wrt_input = torch.sum(self.weights * phi_x_der_mat, dim=1)

#     def compute_derivative_output_wrt_bias(self):
#         # no bias in KAN!
#         self.derivative_output_wrt_bias = 0