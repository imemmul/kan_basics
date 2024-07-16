from .kan import KANNeuron
from .mlp import MLPNeuron
import numpy as np
from tqdm import tqdm
from .loss import SquaredLoss

class Layer:
    def __init__(self, n_inputs, n_outputs, neuron_class=KANNeuron, **kwargs):
        self.n_inputs = n_inputs
        self.neurons = [neuron_class(n_inputs) if (kwargs =={}) else neuron_class(n_inputs, **kwargs) for _ in range(n_outputs)]
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
        self.loss = loss(self.layers_shape[-1])
        
        
    def forward(self, x):
        x_in = x
        for li in range(self.n_layers):
            x_in = self.layers[li].forward(x_in) # forward pass
        return x_in

    def backward(self):
        delta = self.layers[-1].update_grad(self.loss.dloss_dy)
        for ll in range(self.n_layers - 1)[::-1]:
            delta = self.layers[ll].update_grad(delta)

    def update_weights(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.apply_gradient_descent(self.learning_rate)
    
    # def train(self, x_train, y_train, n_iter_max=10000, loss_tol=.05):
    #     self.loss_hist = np.zeros(n_iter_max)
    #     x_train, y_train = np.array(x_train), np.array(y_train)
    #     assert x_train.shape[0] == y_train.shape[0], 'x_train, y_train must contain the same number of samples'
    #     assert x_train.shape[1] == self.layers_shape[0], 'shape of x_train is incompatible with first layer'

    #     pbar = tqdm(range(n_iter_max))
    #     for it in pbar:
    #         loss = 0  # reset loss
    #         for ii in range(x_train.shape[0]):
    #             x_out = self.forward(x_train[ii, :])  # forward pass
    #             loss += self.loss(x_out, y_train[ii, :])  # accumulate loss
    #             self.backward()  # backward propagation
    #             [layer.zero_grad(which=['xin']) for layer in self.layers]  # reset gradient wrt xin to zero
    #         self.loss_hist[it] = loss
    #         if (it % 10) == 0:
    #             pbar.set_postfix_str(f'loss: {loss:.3f}')  #
    #         if loss < loss_tol:
    #             pbar.set_postfix_str(f'loss: {loss:.3f}. Convergence has been attained!')
    #             self.loss_hist = self.loss_hist[: it]
    #             break
    #         self.update_weights()  # update parameters
    #         [layer.zero_grad(which=['weights', 'bias']) for layer in self.layers]  # reset gradient wrt par to zero
    
    def train(self, x_train, y_train, epochs):
        x_train, y_train = np.array(x_train), np.array(y_train)
        assert x_train.shape[0] == y_train.shape[0]
        assert x_train.shape[1] == self.layers_shape[0]
        losses = []
        for epoch in range(epochs):
            
            epoch_loss = 0
            for i in range(x_train.shape[0]):
                
                # Forward pass
                input, target = x_train[i, :], y_train[i, :]
                
                preds = self.forward(input)
                # Compute loss
                epoch_loss += self.loss(preds, target)
                
                self.backward()
                [layer.zero_grad(which=['xin']) for layer in self.layers]  # reset gradient wrt xin to zero
                
            epoch_loss /= len(x_train)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
            self.update_weights()
            [layer.zero_grad(which=['weights', 'bias']) for layer in self.layers]
            losses.append(self.loss.loss)
        return losses