import numpy as np
from scipy.interpolate import BSpline
# from act import silu


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