import math

def silu(x, get_derivative=False):
    """
    SiLU (Sigmoid Linear Unit) activation function.
    
    Parameters:
        x (numpy.ndarray): Input array.
        get_derivative (bool): If True, returns both the SiLU function and its derivative.
    
    Returns:
        numpy.ndarray: SiLU function values.
        numpy.ndarray (optional): Derivative of the SiLU function.
    """
    silu_value = x / (1 + math.exp(-x))
    
    if get_derivative:
        sigmoid = 1 / (1 + math.exp(-x))
        silu_derivative = sigmoid * (1 + x * (1 - sigmoid))
        return silu_value, silu_derivative
    
    return silu_value


def relu(x, get_derivative=False):
    return x * (x > 0) if not get_derivative else 1.0 * (x >= 0)
 
def tanh_act(x, get_derivative=False):
    if not get_derivative:
        return math.tanh(x)
    return 1 - math.tanh(x) ** 2
 
def func_sigmoid(x, get_derivative=False):
    if not get_derivative:
        return 1 / (1 + math.exp(-x))
    return func_sigmoid(x) * (1 - func_sigmoid(x))