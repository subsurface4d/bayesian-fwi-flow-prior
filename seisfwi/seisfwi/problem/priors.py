import torch
import pyro.distributions as dist

import seisfwi.defaults as defaults


def UniformPrior(lower_bound, high_bound, ndim):
    """ Uniform prior distribution for the model parameters
    
    Parameters
    ----------
    lower_bound : array-like
        The lower bound of the uniform distribution.
    high_bound : array-like
        The upper bound of the uniform distribution.
    ndim : int
        The number of dimensions of the model parameters.
    Returns
    -------
    dist.Uniform
        The uniform prior distribution for the model parameters.
    """
    
    if lower_bound.shape != (ndim,) or high_bound.shape != (ndim,):
        raise ValueError(f"lower_bound and high_bound should have shape ({ndim},)")

    low = torch.as_tensor(lower_bound, dtype=defaults.dtype, device=defaults.device)
    high = torch.as_tensor(high_bound, dtype=defaults.dtype, device=defaults.device)

    return dist.Uniform(low, high).to_event(1)


def GaussianPrior(mean, std, ndim):
    """Gaussian prior distribution for the model parameters
    
    Parameters
    ----------
    mean : float
        The mean value of the Gaussian distribution.
    std : float
        The standard deviation of the Gaussian distribution.
    ndim : int
        The number of dimensions of the model parameters.        
    Returns
    -------
    dist.Normal
        The Gaussian prior distribution for the model parameters.
    """
    
    if mean.shape != (ndim,) or std.shape != (ndim,):
        raise ValueError(f"mean and std should have the shape ({ndim},), got {mean.shape} and {std.shape}")

    # Convert to torch tensors safely
    mean = torch.as_tensor(mean, dtype=defaults.dtype, device=defaults.device)
    std  = torch.as_tensor(std, dtype=defaults.dtype, device=defaults.device)

    # Return the multivariate prior
    return dist.Normal(mean, std).to_event(1)
