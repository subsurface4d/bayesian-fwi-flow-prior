import numpy as np
import torch


def norm_traces(data):
    """ Normalize a seismic data to its maximum amplitude trace by trace
    
    Args:
        data (ndarray): The seismic data to be normalized.
        
    Returns:
        ndarray: The normalized seismic data.
    """

    data = data.copy()
    eps = 1e-20

    nr = data.shape[0]
    nt = data.shape[1]


    for i in range(nr):
        data[i,:] = data[i,:]/(abs(data[i,:]) + eps).max()

    return data



def nrms(trace1, trace2):
    """
    Compute the normalized root mean square (NRMS) between two seismic traces
    NRMS = 2 * sqrt(mean((trace2 - trace1)^2)) / (sqrt(mean(trace1^2)) + sqrt(mean(trace2^2))
    :param trace1: seismic trace 1
    :param trace2: seismic trace 2
    :return: NRMS between the two seismic traces
    """
    
    denominator = np.sqrt(np.mean(trace1**2)) + np.sqrt(np.mean(trace2**2))

    if denominator == 0:
        return 0.0  # Avoid division by zero

    numerator = 2 * np.sqrt(np.mean((trace2 - trace1)**2))

    return  numerator / denominator