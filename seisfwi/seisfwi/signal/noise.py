import numpy as np

def add_noise(data, noise, snr=20):
    """
    Add realistic noise to simulated seismic data.
    
    Parameters:
        data (numpy.ndarray): Simulated data array of shape (nrec, nt)
        noise (numpy.ndarray): Noise array of shape (nrec, nt)
        snr (float): Signal-to-noise ratio in dB    
    Returns:
        numpy.ndarray: Noisy data array of shape (nrec, nt)
    """
 
    # Normalize noise to match SNR
    signal_power = np.mean(data**2)
    noise_power = np.mean(noise**2)
    noise = noise * np.sqrt(signal_power / (10**(snr / 10) * noise_power))
    
    # Add noise to data
    noisy_data = data.copy() + noise.copy()
    

    return noisy_data


def compute_snr(data_clean, noise):
    """Compute SNR (dB) with the same convention as add_noise()."""
    signal_power = np.mean(data_clean**2)
    noise_power  = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)
