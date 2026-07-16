import scipy
import warnings
import numpy as np
import torch

def bandpass_filter(data, dt, freqmin, freqmax, order=4):
    """
    Bandpass filter a 2D array along the second axis.
    
    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array of shape (ntraces, nsamples).
    freqmin : float
        Low corner frequency for bandpass filter.
    freqmax : float
        High corner frequency for bandpass filter.
    df : float
        Sampling rate of the data.
    order : int, optional
        Number of order for the filter. Default is 4.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of shape (ntraces, nsamples) containing the
        bandpass filtered data.
        
    Note: Developed by Fu Yin at Rice University.
    """
    
    reshape = False
    
    if data.ndim != 2:
        
        # reshape the data into 2D array
        data = data.reshape(1, -1)
        
        reshape = True
        # msg = "Only 2D arrays are supported."
        # raise ValueError(msg)    
    
    df = 1.0 / dt
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = (
            "Selected high corner frequency ({}) of bandpass is at or "
            "above Nyquist ({}). Applying a high-pass instead."
        ).format(freqmax, fe)
        warnings.warn(msg)
        
        raise ValueError(msg)
    
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        
        raise ValueError(msg)
    
    z, p, k = scipy.signal.iirfilter(
        order, [low, high], btype="band", ftype="butter", output="zpk"
    )
    
    sos = scipy.signal.zpk2sos(z, p, k)
    
    ### obspy style
    firstpass = scipy.signal.sosfilt(sos, data, axis=1)
    out = scipy.signal.sosfilt(sos, firstpass[:, ::-1], axis=1)[:, ::-1]
    
    if reshape:
        out = out.reshape(-1)
    
    return out.astype(data.dtype)
