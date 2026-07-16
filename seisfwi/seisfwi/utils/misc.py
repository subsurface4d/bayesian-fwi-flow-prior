import numpy as np
import pandas as pd

from io import StringIO
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn



def smooth2d(data, span_x, span_z):
    ''' Smooths values 2D rectangular grid
    '''

    return gaussian_filter(data, sigma=(span_z, span_x))


def interp_data2d(data, dx=12.5, dz=2.5, dh = 10, method='linear'):
    
    nx, nz = data.shape
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz

    xi = np.linspace(x[0], x.max(), int(round((x.max() - x[0]) / dh)) + 1)
    zi = np.linspace(z[0], z.max(), int(round((z.max() - z[0]) / dh)) + 1)

    xx, zz = np.meshgrid(xi, zi, indexing='ij')

    return interpn((x, z), data, np.array([xx, zz]).T, method = method).T


def interp_data3d(data, dx=12.5, dy=12.5, dz=2.5, dh=10, method='linear'):
    # Input validation
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("Input data must be a 3D numpy array.")

    # Original grid dimensions
    nx, ny, nz = data.shape
    x = np.arange(0, nx*dx, dx)
    y = np.arange(0, ny*dy, dy)
    z = np.arange(0, nz*dz, dz)

    # New grid
    xi = np.arange(0, x.max(), dh)
    yi = np.arange(0, y.max(), dh)
    zi = np.arange(0, z.max(), dh)
    xx, yy, zz = np.meshgrid(xi,yi,zi, indexing='ij')
    
    return interpn((x,y,z), data, np.array([xx, yy, zz]).T, method = method)



def load_misfit(misfit_file):
    ''' Load the misfit from the file
    '''

    iters = []
    misfit = []
    with open(misfit_file) as f:
        for line in f:
            values = line.split()
            try:
                iters.append(int(values[0]))
                misfit.append(float(values[1]))
            except:
                pass

    data = np.column_stack((iters, misfit))
    smallest_values = [np.min(data[data[:, 0] == index][:, 1]) 
                       for index in np.unique(data[:, 0])]

    smallest_values.insert(0, misfit[0])
    
    return np.array(smallest_values)


def load_log_file(log_file):
    """
    Load iteration info from an optimizer log file.

    Parameters
    ----------
    log_file : str or Path
        Path to the log file.
    col_names : list of str, optional
        Column names for the output DataFrame.

    Returns
    -------
    DataFrame
        Iteration table.
    """

    with open(log_file, 'r') as f:
        data_lines = [
            line.strip()
            for line in f
            if line.strip() and line.lstrip()[0].isdigit()
        ]

    if not data_lines:
        raise ValueError('No iteration lines found in log file.')

    # this is fixed 
    col_names = ['Niter', 'fk', 'gk_norm', 'fk_f0', 'alpha', 'nls', 'ngrad']

    df = pd.read_csv(
        StringIO('\n'.join(data_lines)),
        sep=r'\s+',
        header=None,
        names=col_names,
        skipinitialspace=True
    )

    return df
