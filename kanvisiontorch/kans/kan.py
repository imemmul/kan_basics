import torch
from torch import nn
import torch.nn.functional as F
from typing import *
from .utils import L1
from torch.profiler import profile, record_function


# NOTE nn.GELU fixed to conv 0 gradient problem
class KAN(nn.Module):
    def __init__(self, layers_shape, k=3, device='cpu', base_activation=nn.GELU,
                x_bounds=[-1,1], **kwargs):
        super(KAN, self).__init__()
        self.layers_shape = layers_shape
        self.k = k
        self.base_activation = base_activation
        self.x_bounds = x_bounds
        self.layers = nn.ModuleList()
        self.n_layers = len(layers_shape[:-1])
        for i in range(self.n_layers):
            self.layers.append(BSplineLayer(n_in=layers_shape[i], n_out=layers_shape[i+1],
                                        base_activation=base_activation, x_bounds=x_bounds, k=k, **kwargs))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = x.view(x.size(0), -1)
            x = layer(x)
        return x

# TODO implement piecewise below doesn't work well
class PieceWiseLinearIntLayer(nn.Module):
    def __init__(self, n_in, n_out, base_activation=nn.GELU, device='cuda', **kwargs):
        super(PieceWiseLinearIntLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        self.base_activation = base_activation()
        self.weights = nn.Parameter(torch.randn(n_out, n_in, device=self.device))
        self.biases = nn.Parameter(torch.randn(n_out, device=self.device))
        
        self.grid_points = nn.Parameter(torch.linspace(-1, 1, 10, device=self.device), requires_grad=False)
        self.slopes = nn.Parameter(torch.randn(n_out, n_in, len(self.grid_points) - 1, device=self.device))
        self.intercepts = nn.Parameter(torch.randn(n_out, n_in, len(self.grid_points) - 1, device=self.device))
        
        self.layer_norm = nn.LayerNorm(n_out, elementwise_affine=False).to(self.device)
        self.prelu = nn.PReLU().to(self.device)
        nn.init.kaiming_uniform_(self.weights, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.slopes, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.intercepts, nonlinearity='linear')
    
    def forward(self, x):
        x = x.to(self.device)
        b_out = F.linear(self.base_activation(x), self.weights, self.biases)
        
        num_samples = x.size(0)
        x_expanded = x.unsqueeze(-1)
        diff = x_expanded - self.grid_points
        pos = (diff >= 0).float()

        segment_index = pos.sum(dim=-1).long() - 1
        segment_index = torch.clamp(segment_index, 0, len(self.grid_points) - 2)
        
        # getting the slopes and intercepts for each segment
        segment_index_expanded = segment_index.unsqueeze(-1).unsqueeze(-1).expand(num_samples, self.n_in, 1, self.n_out)
        slopes = torch.gather(self.slopes.permute(2, 1, 0).unsqueeze(0).expand(num_samples, -1, -1, -1), 1, segment_index_expanded).squeeze(2).permute(0, 2, 1)
        intercepts = torch.gather(self.intercepts.permute(2, 1, 0).unsqueeze(0).expand(num_samples, -1, -1, -1), 1, segment_index_expanded).squeeze(2).permute(0, 2, 1)
        
        # Apply the piecewise linear transformation
        x_expanded = x.unsqueeze(1).expand(num_samples, self.n_out, self.n_in)
        linear_output = slopes * x_expanded + intercepts
        
        # Sum over the segments
        linear_output = linear_output.sum(dim=-1)
        
        x = self.prelu(self.layer_norm(b_out + linear_output))
        return x
    
class BSplineLayer(nn.Module):
    def __init__(self, n_in, n_out, base_activation=nn.GELU, x_bounds=[-1, 1], k=3, device='cpu', **kwargs):
        super(BSplineLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        self.base_activation = base_activation()
        # constant grid size decreases loss and increases accuracy, why ??
        # self.grid_size = self.n_out - k + 1
        self.grid_size = 5
        self.k = k  # spline order
        self.x_bounds = x_bounds
        
        self.weights = nn.Parameter(torch.randn((n_out, n_in), device=self.device, dtype=torch.float32))
        self.spline_weights = nn.Parameter(torch.randn((n_out, n_in, self.grid_size + k), device=self.device, dtype=torch.float32))
        
        self.layer_norm = nn.LayerNorm(self.n_out, elementwise_affine=False)
        self.prelu = nn.PReLU(device=self.device)
        
        step = (self.x_bounds[1] - self.x_bounds[0]) / self.grid_size
        self.grid_values = torch.linspace(self.x_bounds[0] - step * k,
                                          self.x_bounds[1] + step * k,
                                          self.grid_size + 2 * self.k + 1,
                                          dtype=torch.float32).expand(self.n_in, -1).contiguous().to(self.device)
        
        nn.init.kaiming_uniform_(self.weights, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.spline_weights, nonlinearity='linear')
    
    def forward(self, x):
        grid_values = self.grid_values.to(x.device)
        b_out = F.linear(self.base_activation(x), self.weights)
        x_uns = x.unsqueeze(-1)
        bases = ((x_uns >= grid_values[:, :-1]) & (x_uns < grid_values[:, 1:])).to(x.dtype).to(x.device)
        for k in range(1, self.k + 1):
            left_interval = grid_values[:, :-(k + 1)]
            right_interval = grid_values[:, k:-1]
            delta = torch.where(right_interval == left_interval, torch.ones_like(right_interval), right_interval - left_interval)
            bases = ((x_uns - left_interval) / delta) * bases[:, :, :-1] + \
                    ((grid_values[:, k + 1:] - x_uns) / (grid_values[:, k + 1:] - grid_values[:, 1:-k]) * bases[:, :, 1:])
        # print(f"Bases shape: {bases.shape}, bases:{torch.sum(bases) == 0}")
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