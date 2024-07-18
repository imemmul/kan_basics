import torch
import torch.nn.functional as F

def silu(x, get_derivative=False):
    """
    SiLU (Sigmoid Linear Unit) activation function.

    Parameters:
        x (torch.Tensor): Input tensor.
        get_derivative (bool): If True, returns both the SiLU function and its derivative.

    Returns:
        torch.Tensor: SiLU function values.
        torch.Tensor (optional): Derivative of the SiLU function.
    """
    silu_value = x * torch.sigmoid(x)
    
    if get_derivative:
        sigmoid = torch.sigmoid(x)
        silu_derivative = sigmoid * (1 + x * (1 - sigmoid))
        return silu_value, silu_derivative
    
    return silu_value

def softmax(logits):
    """
    Softmax function.

    Parameters:
        logits (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Softmax function values.
    """
    exp_logits = torch.exp(logits - torch.max(logits, dim=-1, keepdim=True)[0])
    return exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

def relu(x, get_derivative=False):
    """
    ReLU activation function.

    Parameters:
        x (torch.Tensor): Input tensor.
        get_derivative (bool): If True, returns the derivative of the ReLU function.

    Returns:
        torch.Tensor: ReLU function values.
        torch.Tensor (optional): Derivative of the ReLU function.
    """
    if get_derivative:
        return (x > 0).float()
    return F.relu(x)

def tanh_act(x, get_derivative=False):
    """
    Tanh activation function.

    Parameters:
        x (torch.Tensor): Input tensor.
        get_derivative (bool): If True, returns the derivative of the Tanh function.

    Returns:
        torch.Tensor: Tanh function values.
        torch.Tensor (optional): Derivative of the Tanh function.
    """
    if not get_derivative:
        return torch.tanh(x)
    return 1 - torch.tanh(x) ** 2

def sigmoid(x, get_derivative=False):
    """
    Sigmoid activation function.

    Parameters:
        x (torch.Tensor): Input tensor.
        get_derivative (bool): If True, returns the derivative of the Sigmoid function.

    Returns:
        torch.Tensor: Sigmoid function values.
        torch.Tensor (optional): Derivative of the Sigmoid function.
    """
    if not get_derivative:
        return torch.sigmoid(x)
    sig = torch.sigmoid(x)
    return sig * (1 - sig)
