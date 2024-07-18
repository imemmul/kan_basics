import torch
from torch import tensor
from activations import silu
from scipy.interpolate import BSpline
import numpy as np

def get_bsplines(x_bounds, n_basis, k=3, **kwargs):
    """
    •   The function get_bsplines generates B-spline basis functions and their derivatives within the specified bounds x_bounds.
	•	It constructs a knot vector and initializes special cases (like the SiLU function for the 0th basis function).
	•	For each basis function, it creates the B-spline basis function using the specified knots and calculates its derivative.
	•	The function returns dictionaries edge_fun and edge_fun_der containing the B-spline basis functions and their derivatives, respectively.
    """
    grid_len = n_basis - k + 1
    step = x_bounds[1] - x_bounds[0] / (grid_len - 1)
    edge_fun, edge_fun_der = {}, {}
    
    # silu bias
    edge_fun[0] = lambda x: x / (1 + np.exp(-x))
    edge_fun_der[0] = lambda x: (1 + np.exp(-x) + x * np.exp(-x)) / np.power((1 + np.exp(-x)), 2)
    
    t = np.linspace(x_bounds[0] - k * step, x_bounds[1] + k * step, grid_len + 2 * k)
    t[k] , t[-k - 1] = x_bounds[0], x_bounds[1]
    
    for ind_spline in range(n_basis - 1):
        edge_fun[ind_spline + 1] = BSpline.basis_element(t[ind_spline:ind_spline + k + 2], extrapolate=False)
        edge_fun_der[ind_spline + 1] = edge_fun[ind_spline + 1].derivative()
    
    return edge_fun, edge_fun_der

def get_bsplines_torch(x_bounds, n_basis, k=3, device='cuda', **kwargs):
    """
    Generate B-spline basis functions and their derivatives within the specified bounds x_bounds.
    The function constructs a knot vector and initializes special cases (like the SiLU function for the 0th basis function).
    For each basis function, it creates the B-spline basis function using the specified knots and calculates its derivative.
    The function returns dictionaries edge_fun and edge_fun_der containing the B-spline basis functions and their derivatives, respectively.
    """
    grid_len = n_basis - k + 1
    step = (x_bounds[1] - x_bounds[0]) / (grid_len - 1)
    edge_fun, edge_fun_der = {}, {}
    
    # Silu bias
    edge_fun[0] = lambda x: silu(x)
    edge_fun_der[0] = lambda x: silu(x, get_derivative=True)[1]
    
    t = np.linspace(x_bounds[0] - k * step, x_bounds[1] + k * step, grid_len + 2 * k)
    t[k], t[-k - 1] = x_bounds[0], x_bounds[1]
    
    for ind_spline in range(n_basis - 1):
        bspline = BSpline.basis_element(t[ind_spline:ind_spline + k + 2], extrapolate=False)
        edge_fun[ind_spline + 1] = lambda x, bspline=bspline: torch.tensor(bspline(x.cpu().numpy()), device=device, dtype=torch.float32)
        edge_fun_der[ind_spline + 1] = lambda x, bspline=bspline: torch.tensor(bspline.derivative()(x.cpu().numpy()), device=device, dtype=torch.float32)
    
    return edge_fun, edge_fun_der