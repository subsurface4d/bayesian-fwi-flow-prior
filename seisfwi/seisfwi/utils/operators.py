import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
from seisfwi.defaults import device, dtype

# ---------------------- Gradient Mask ----------------------

class GradientMaskFunction(Function):
    """
    Custom autograd Function that applies a gradient mask
    to the backward pass only.
    """
    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(mask)
        # Do nothing in forward pass, just return input
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        masked_grad = mask * grad_output
        return masked_grad, None


class MaskOperator(torch.nn.Module):
    """
    A module that applies a mask to the gradients during backward pass.

    Parameters
    ----------
    mask : array_like or torch.Tensor
        Mask array broadcastable to input shape.
    """
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, input):
        return GradientMaskFunction.apply(input, self.mask)

# ---------------------- Gradient Smoother ----------------------

class SmoothGradientFunction(Function):
    """
    Custom autograd Function that applies a smoothing kernel
    to the gradient during backward pass.
    """
    @staticmethod
    def forward(ctx, input, kernel):
        ctx.save_for_backward(kernel)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        kernel, = ctx.saved_tensors
        kx, kz = kernel.shape

        grad_input = F.conv2d(
            grad_output.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=(kx // 2, kz // 2)
        )
        return grad_input.squeeze(0).squeeze(0), None


class SmoothOperator(torch.nn.Module):
    """
    Apply 2D Gaussian smoothing to gradients during backward pass.

    Parameters
    ----------
    kernel_size_x : int
        Kernel size in x-direction.
    kernel_size_z : int
        Kernel size in z-direction.
    sigma : float, optional
        Standard deviation for Gaussian kernel.
    """
    def __init__(self, kernel_size_x, kernel_size_z, sigma=2):
        super().__init__()
        # swap kernel axes
        kx, kz = kernel_size_z, kernel_size_x

        if kx % 2 == 0 or kz % 2 == 0:
            raise ValueError("Kernel sizes must be odd integers.")

        x = torch.linspace(-sigma, sigma, steps=kx)
        z = torch.linspace(-sigma, sigma, steps=kz)
        gx = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gz = torch.exp(-z.pow(2) / (2 * sigma ** 2))
        kernel = torch.outer(gx, gz) / (2 * np.pi * sigma ** 2)
        kernel /= kernel.sum()
        
        self.kernel = kernel.to(dtype=dtype, device=device)

    def forward(self, input):
        return SmoothGradientFunction.apply(input, self.kernel)



def ConstrainOperator(value: torch.Tensor, 
                      min_value: float, 
                      max_value: float) -> torch.Tensor:
    """
    Clamp tensor values within [min_value, max_value].

    Parameters
    ----------
    value : torch.Tensor
        Input tensor to be constrained.
    min_value : float
        Minimum allowed value.
    max_value : float
        Maximum allowed value.

    Returns
    -------
    torch.Tensor
        Clamped tensor.
    """
    if min_value is not None and max_value is not None:
        value = torch.clamp(value, min_value, max_value)
        _assert_finite(value)
    return value


def _assert_finite(tensor: torch.Tensor):
    """
    Utility: Check tensor for NaN or Inf values.
    Raises ValueError if any found.
    """
    if torch.isinf(tensor).any():
        raise ValueError("[ConstrainOperator] Found Inf in tensor.")
    if torch.isnan(tensor).any():
        raise ValueError("[ConstrainOperator] Found NaN in tensor.")

