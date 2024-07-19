import torch
from torch import nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    def __init__(self, n_in, n_weights_per_edge, base_activation=nn.Tanh, x_bounds=[-1, 1], k=3, device='cpu', **kwargs):
        super(KANLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_weights_per_edge
        self.device = device
        self.base_activation = base_activation()
        self.grid_size = self.n_out - k + 1 # FIXME is this correct
        self.k = k # spline order
        self.x_bounds = x_bounds
        
        self.weights = nn.Parameter(torch.rand((self.n_out, self.n_in), device=device, dtype=torch.float32))
        self.spline_weights = nn.Parameter(torch.rand((self.n_out, self.n_in, self.grid_size + self.k), device=device, dtype=torch.float32))
        
        self.layer_norm = nn.LayerNorm(self.n_out, elementwise_affine=False)
        
        #TODO PReLU ???
        self.prelu = nn.PReLU()
        step = (self.x_bounds[1] - self.x_bounds[0]) / self.grid_size
        self.grid_values = torch.linspace(self.x_bounds[0] - step * self.k,
                                          self.x_bounds[1] + step * self.k,
                                          self.grid_size + 2 * self.k + 1,
                                          dtype=torch.float32).expand(self.n_in, -1).contiguous().to(device)
        
        nn.init.kaiming_uniform_(self.weights, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.spline_weights, nonlinearity='linear')
    
    def forward(self, x):
        grid_values = self.grid_values
        b_out = F.linear(self.base_activation(x), self.weights)
        # print(f"x: {x.shape}, b_out: {b_out.shape}")
        x = x.unsqueeze(-1)
        bases = ((x >= grid_values[:, :-1]) & (x < grid_values[:, 1:])).to(x.dtype).to(self.device)
        for k in range(1, self.k + 1):
            left_interval = grid_values[:, :-(k+1)]
            right_interval = grid_values[:, k:-1]
            delta = torch.where(right_interval == left_interval, torch.ones_like(right_interval), right_interval - left_interval)
            bases = ((x - left_interval) / delta) * bases[:, :, :-1] + \
                    ((grid_values[:, k + 1:] - x) / (grid_values[:, k + 1:] - grid_values[:, 1:-k]) * bases[:, :, 1:])
        bases = bases.contiguous()
        spline_output = F.linear(bases.view(x.size(0), -1), self.spline_weights.view(self.spline_weights.size(0), -1))
        x = self.prelu(self.layer_norm(b_out + spline_output))
        return x 

if __name__ == "__main__":
    # testing KANLayer
    kan_layer = KANLayer(3, 7)
    x = torch.rand((2, 3))
    y = kan_layer(x)
    print(y.shape)
    