import numpy as np
import matplotlib.pyplot as plt
from seisfwi.signal import norm_traces


def plot_data(data, dt=None, dx=None, time=None, offset=None, cmap='gray', aspect='auto', clip=99.9, 
                figsize=(10,6), colorbar=False, savefig=None, 
                type='section', norm=False, wiggle_scale=1, wiggle_interval=1,
                plot_fk=False, vel=None, fmin=1, fmax=20, kmin=-0.05, kmax=0.05):
    """ Plot two data
    
    Plot a data section using matplotlib.imshow.
    
    Parameters:
    - data (ndarray or Tensor): The data to be plotted.
    - dt (float, optional): The time sampling interval. If not provided, the x-axis will be labeled with sample numbers.
    - dx (float, optional): The spatial sampling interval. If not provided, the y-axis will be labeled with trace numbers.
    - offset (float, optional): The offset of the data. If provided, the x-axis will be labeled with offset values.
    - cmap (str, optional): The colormap to be used. Default is 'gray'.
    - aspect (str, optional): The aspect ratio of the plot. Default is 'auto'.
    - clip (float, optional): The percentile value for clipping the data. Default is 99.9.
    - figsize (tuple, optional): The size of the figure. Default is (8, 8).
    - colorbar (bool, optional): Whether to show the colorbar. Default is False.
    - norm (bool, optional): Whether to normalize the data. Default is False.
    - savefig (str, optional): The file path to save the figure. If not provided, the figure will be displayed.

    Returns:
    None
    """

    plt.figure(figsize = figsize)
    
    if not isinstance(data, np.ndarray):
        try:
            data = data.cpu().detach().numpy()
        except AttributeError:
            # Handle the case where data cannot be converted to a NumPy array
            pass    
    if dt is None:
        t = np.arange(data.shape[1])
    else:
        t = np.arange(data.shape[1]) * dt
    if dx is None:
        x = np.arange(data.shape[0])
    else:
        x = np.arange(data.shape[0]) * dx

    if time is not None:
        t = time

    if offset is not None:
        x = offset

    extent = [x[0], x[-1], t[-1], t[0]]

    if plot_fk:
        plt.subplot(1,2,1)
    else:
        plt.subplot(1,1,1)

    if norm:
        data = norm_traces(data)

    vmax = np.percentile(data, clip)
    if type == 'section':
        plt.imshow(data.T, aspect=aspect, cmap=cmap, vmin=-vmax, vmax=vmax, extent = extent)
        plt.ylim([t[-1], t[0]])
        plt.xlim([x[0], x[-1]])
    
    elif type == 'wiggle':
        for i, trace in enumerate(data):
            if i % wiggle_interval != 0:
                continue
            trace = trace * wiggle_scale + i
            plt.plot(trace, t, color='black', linewidth=1.0)
            plt.fill_betweenx(t, i, trace, where=(trace > i), color='black')
        plt.ylim([t[-1], t[0]])
        plt.xlim([0-wiggle_scale, data.shape[0]+wiggle_scale])
    else:
        raise ValueError('type must be either "section" or "wiggle"')

    # lim

    if dx is not None:
        plt.xlabel('Offset (m)')
    else:
        plt.xlabel('Trace #')

    if dt is not None:
        plt.ylabel('Time (s)')
    else:
        plt.ylabel('Sample #')
    if colorbar:
        plt.colorbar()
    plt.grid(axis='y', alpha=0.8)

    # if plot_fk:
        
    #     if dx is None or dt is None:
    #         raise ValueError("dx and dt must be provided to plot the f-k domain")

    #     fk, fft_f, fft_k = fk_numpy(data, dx, dt)
    #     plt.subplot(1,2,2)
    #     plt.imshow(fk.T, extent=[fft_k[0], fft_k[-1], fft_f[-1], fft_f[0]], aspect="auto")
    #     if vel is not None:
    #         for v in vel:
    #             plt.plot(fft_k, v * fft_k, 'w--')
    #             plt.plot(fft_k, -v * fft_k, 'w--')

    #     plt.ylim([fmin, fmax])
    #     plt.xlim([kmin, kmax])
    #     plt.ylabel('Frequency (Hz)')
    #     plt.xlabel('Wavenumber (1/m)')
    #     plt.grid(True, axis='y', alpha=0.5)
    #     plt.gca().set_aspect('auto', adjustable='box')
    #     plt.colorbar()

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi=300)

    plt.show()
    plt.close()



def plot_trace(trace, dt=None, figsize=(8, 3), figsave=None):
    """
    Plots a wiggle plot for seismic data.

    Parameters:
    trace (1D numpy array): The seismic trace data.
    t (1D numpy array, optional): The time values corresponding to each sample in the trace.
        If not provided, the sample number will be used as the x-axis.
    dt (float, optional): The time interval between samples in the trace.
        If provided, the time values will be calculated using the sample number and dt.
    figsize (tuple, optional): The size of the figure in inches (width, height).
    figsave (str, optional): The file path to save the figure.
        If provided, the figure will be saved instead of displayed.

    Returns:
    None
    """
    if dt is not None:
        t = np.arange(trace.shape[0]) * dt
        xlabel = 'Time (s)'
    else:
        t = np.arange(trace.shape[0])
        xlabel = 'Sample #'

    plt.figure(figsize=figsize)
    plt.plot(t, trace, color='black', linewidth=2.0)
    plt.fill_between(t, trace, 0, where=(trace > 0), color='black')
    plt.xlabel(xlabel)
    plt.ylabel('Amplitude')
    
    max = np.max(np.abs(trace))
    plt.ylim([-max*1.1, max*1.1])
    plt.grid(alpha=0.5, linestyle='--')
    
    # plt.gca().invert_yaxis()
    plt.tight_layout()

    if figsave:
        plt.savefig(figsave, dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()
        
        
def plot_stf(t, stf, fmax=50, figsize=(10, 3), figname=None, amp_ylim=None, fre_ylim=None):
    ''' Plot source time function
    '''

    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.plot(t, stf, 'k-')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Source Time Function')
    plt.grid(True)
    plt.xlim(t[0], t[-1])
    if amp_ylim is not None:
        plt.ylim(amp_ylim)
    
    plt.subplot(1,2,2)
    dt = t[1] - t[0]
    n = len(t)
    freq = np.fft.fftfreq(n, dt)
    spectrum = np.abs(np.fft.fft(stf))
    idx = np.argsort(freq)
    idx = idx[int(len(idx) / 2):]
    
    plt.plot(freq[idx], spectrum[idx], 'k-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum')
    plt.grid(True)
    plt.xlim(0, fmax)
    if fre_ylim is not None:
        plt.ylim(fre_ylim)
        
    plt.tight_layout()

    if figname is not None:
        plt.savefig(figname, dpi=300)

    plt.show()
    plt.close()


def plot_misfit(misfits, label=None, semilogy=True, normalize=True, figsize=(8, 5), figname=None):
    ''' Plot misfit
    '''

    if type(misfits) is not list:
        misfits= [misfits]

    plt.figure(figsize=figsize)
    for i, misfit in enumerate(misfits):
        if normalize:
            misfit = misfit / abs(misfit[0])

        if semilogy:
            if label is not None:
                plt.semilogy(misfit,  marker='o', linestyle='-', label=label[i])
            else:
                plt.semilogy(misfit,  marker='o', linestyle='-',)        
        else:
            if label is not None:
                plt.plot(misfit,  marker='o', linestyle='-', label=label[i])
            else:
                plt.plot(misfit,  marker='o', linestyle='-')
    if label is not None:
        plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Misfit')
    plt.title('Misfit')
    plt.grid(True)
    plt.tight_layout()

    if figname is not None:
        plt.savefig(figname, dpi=300)
        
    plt.show()
    plt.close()
