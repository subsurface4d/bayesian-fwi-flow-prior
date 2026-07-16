import numpy as np

def wavelet(nt, dt, f0, amp0 = 1, t0 = None, type = 'Ricker'):
    """ source time function.
    """
    # time array
    t = np.arange(nt) * dt + 0.0
    wavelet = np.zeros_like(t)
    t0 = t0 if t0 is not None else 1.2 / f0

    # Ricker wavelet
    if type.lower() in ['ricker']:
        temp = (np.pi*f0) ** 2
        wavelet = amp0 * (1 - 2 * temp * (t - t0) ** 2) * np.exp(- temp * (t - t0) ** 2)
    
    # Gaussian wavelet
    elif type.lower() in ['gaussian']:
        temp = (np.pi*f0) ** 2
        wavelet = amp0 * (1 - 2 * temp * (t - t0) ** 2) * np.exp(- temp * (t - t0) ** 2)

        # perform integration twice to get the Gaussian wavelet
        wavelet = np.cumsum(wavelet)
        wavelet = np.cumsum(wavelet)

    # Ramp wavelet
    elif type.lower() in ['ramp']:
        wavelet = amp0 * 0.5 * (1. + np.tanh(t / t0))
        
    # Unknown source type
    else:
        msg = 'Support source types: Rikcer, Guassian, Ramp. \n'
        err = 'Unknown source type: {}'.format(type)
        raise ValueError(msg + '\n' + err)

    return wavelet
