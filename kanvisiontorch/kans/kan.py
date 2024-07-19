import torch
from torch import nn
import torch.nn.functional as F

class KAN(nn.Module):
    def __init__(self, layers_shape, k=3, device='cpu', base_activation=nn.Tanh,
                x_bounds=[-1,1], **kwargs):
        super(KAN, self).__init__()
        self.layers_shape = layers_shape
        self.k = k
        self.base_activation = base_activation
        self.x_bounds = x_bounds
        self.layers = nn.ModuleList()
        self.n_layers = len(layers_shape[:-1])
        # print(f"n_layers: {self.n_layers}")
        for i in range(self.n_layers):
            # print(layers_shape[i], layers_shape[i+1])
            self.layers.append(KANLayer(n_in=layers_shape[i], n_out=layers_shape[i+1],
                                        base_activation=base_activation, x_bounds=x_bounds, k=k, device=device, **kwargs))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KANLayer(nn.Module):
    def __init__(self, n_in, n_out, base_activation=nn.Tanh, x_bounds=[-1, 1], k=3, device='cpu', **kwargs):
        super(KANLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        self.base_activation = base_activation()
        self.grid_size = self.n_out - k + 1 # FIXME is this correct
        # self.grid_size = 5
        self.k = k # spline order
        self.x_bounds = x_bounds
        
        self.weights = nn.Parameter(torch.rand((self.n_out, self.n_in), device=self.device, dtype=torch.float32))
        self.spline_weights = nn.Parameter(torch.rand((self.n_out, self.n_in, self.grid_size + self.k), device=self.device, dtype=torch.float32))
        
        self.layer_norm = nn.LayerNorm(self.n_out, elementwise_affine=False)
        
        #TODO PReLU ???
        self.prelu = nn.PReLU(device=self.device)
        step = (self.x_bounds[1] - self.x_bounds[0]) / self.grid_size
        self.grid_values = torch.linspace(self.x_bounds[0] - step * self.k,
                                          self.x_bounds[1] + step * self.k,
                                          self.grid_size + 2 * self.k + 1,
                                          dtype=torch.float32).expand(self.n_in, -1).contiguous().to(self.device)
        
        nn.init.kaiming_uniform_(self.weights, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.spline_weights, nonlinearity='linear')
    
    def forward(self, x):
        grid_values = self.grid_values.to(x.device)
        b_out = F.linear(self.base_activation(x), self.weights)
        # print(f"x: {x.shape}, b_out: {b_out.shape}")
        x = x.unsqueeze(-1)
        bases = ((x >= grid_values[:, :-1]) & (x < grid_values[:, 1:])).to(x.dtype).to(x.device)
        for k in range(1, self.k + 1):
            left_interval = grid_values[:, :-(k+1)]
            right_interval = grid_values[:, k:-1]
            delta = torch.where(right_interval == left_interval, torch.ones_like(right_interval), right_interval - left_interval)
            # delta = delta.to(x.device)
            bases = ((x - left_interval) / delta) * bases[:, :, :-1] + \
                    ((grid_values[:, k + 1:] - x) / (grid_values[:, k + 1:] - grid_values[:, 1:-k]) * bases[:, :, 1:])
        bases = bases.contiguous()
        spline_output = F.linear(bases.view(x.size(0), -1), self.spline_weights.view(self.spline_weights.size(0), -1))
        x = self.prelu(self.layer_norm(b_out + spline_output))
        return x 

if __name__ == "__main__":
    # testing KANLayer
    kan_layer = KAN(layers_shape=[3, 3, 3, 1])
    x = torch.rand((2, 3))
    y = kan_layer(x)
    print(y.shape)
    