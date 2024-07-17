import numpy as np


class Neuron:
    def __init__(self, num_inputs, num_weights_per_edge, weights_range=None):
        self.num_inputs = num_inputs
        self.num_weights_per_edge = num_weights_per_edge
        weights_range = [-1, 1] if weights_range is None else weights_range
        self.weights = np.random.uniform(weights_range[0], weights_range[-1], size=(self.num_inputs, self.num_weights_per_edge))
        self.bias = 0
        self.input_values = None
        self.edge_values = None
        self.output_value = None

        self.derivative_output_wrt_edge = None
        self.derivative_output_wrt_bias = None
        self.derivative_edge_wrt_weights = None
        self.derivative_edge_wrt_input = None
        self.derivative_output_wrt_input = None
        self.derivative_output_wrt_weights = None

        self.gradient_loss_wrt_weights = np.zeros((self.num_inputs, self.num_weights_per_edge))
        self.gradient_loss_wrt_bias = 0

    def __call__(self, input_values):
        # print(f"input_values: {input_values}, shape: {np.shape(input_values)}")
        self.input_values = np.array(input_values)
        self.compute_edge_values()
        self.compute_output_value()

        self.compute_derivative_output_wrt_edge()
        self.compute_derivative_output_wrt_bias()
        self.compute_derivative_edge_wrt_weights()
        self.compute_derivative_edge_wrt_input()
        assert self.derivative_output_wrt_edge.shape == (self.num_inputs,)
        # print(f"self.derivative_edge_wrt_input: {self.derivative_edge_wrt_input.shape}, self.num_inputs: {self.num_inputs}")
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
        self.derivative_output_wrt_weights = np.diag(self.derivative_output_wrt_edge) @ self.derivative_edge_wrt_weights

    def update_gradients(self, derivative_loss_wrt_output):
        self.gradient_loss_wrt_weights += self.derivative_output_wrt_weights * derivative_loss_wrt_output
        # print(f"self.derivative_output_wrt_bias: {self.derivative_output_wrt_bias}, derivative_loss_wrt_output: {derivative_loss_wrt_output}")
        self.gradient_loss_wrt_bias += self.derivative_output_wrt_bias * derivative_loss_wrt_output

    def zero_grad(self):
        self.gradient_loss_wrt_weights.fill(0)
        self.gradient_loss_wrt_bias = 0

    def apply_gradient_descent(self, learning_rate):
        self.weights -= learning_rate * self.gradient_loss_wrt_weights
        self.bias -= learning_rate * self.gradient_loss_wrt_bias