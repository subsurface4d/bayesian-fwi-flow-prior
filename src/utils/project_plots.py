import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from tqdm.notebook import tqdm
import matplotlib.animation as animation
from scipy.stats import gaussian_kde

import arviz as az



def configure_plot_settings():
    '''
    Configure the plot settings
    '''
    
    font_paths = [
        '/homes/sep/haipeng/.local/share/fonts/Helvetica.ttf',
        '/homes/sep/haipeng/.local/share/fonts/Helvetica-Bold.ttf',
        '/homes/sep/haipeng/.local/share/fonts/Helvetica-Oblique.ttf',
        '/homes/sep/haipeng/.local/share/fonts/Helvetica-BoldOblique.ttf'
    ]
    for font_path in font_paths:
        fm.fontManager.addfont(font_path)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams.update({'font.size': 12})



def plot_trace(posterior_az, save_path=None):
    
    
    # Generate the trace plot with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), dpi=200, constrained_layout=True)
    axes = np.atleast_2d(axes)  # Ensure axes is 2D

    az.plot_trace(
        posterior_az,
        var_names=['vp'],
        divergences=None,
        kind="trace",
        combined=False,
        legend=False,
        compact=True,
        axes=axes 
    )

    ax1 = axes[0][0]
    ax1.set_title(f"Distribution of RBF Model", fontsize=14) 
    ax1.set_xlabel("Vp (m/s)", fontsize=12)
    ax1.set_ylabel("Counts", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.tick_params(axis="both", which="major", labelsize=10)

    ax2 = axes[0][1]
    ax2.set_title(f"Trace Plot of RBF Model", fontsize=14)
    ax2.set_xlabel("Samples", fontsize=12)
    ax2.set_ylabel("Vp (m/s)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.tick_params(axis="both", which="major", labelsize=10)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    
    
def plot_autocorr(posterior_az, pars):

    # Trace autocorr
    az.plot_autocorr(posterior_az, var_names=['m'], max_lag=pars['num_samples'], 
                     combined=False, figsize=(12,12))
    # plt.savefig(f"{path}/Figures/FWI-HMC-Autocorr.pdf", dpi=300)
    plt.show()
    
    
def plot_posterior(posterior_az, coord = None):
    
    # Trace histograms
    # coords={'vp_dim_0': [0, 1, 2]}
    az.plot_posterior(posterior_az, var_names=['m'], kind='hist', bins=50, coords=None)

    # plt.savefig(f"{path}/Figures/FWI-HMC-Posterior.pdf", dpi=300)
    plt.show()
    
    
def plot_profile(z, vp_true, vp_fwi, vp_mean, vp_p10, vp_p90, vp_map=None, dists = [5, 7.5, 10], save_path=None, xlim=None, ylim=None):    
    
    dx = z[1] - z[0]

    # Create the plot
    fontsize = 14
    plt.figure(figsize=(12, 6), dpi=100)

    for i, dist in enumerate(dists):
        x_ind = np.int64(dist/dx)
        plt.subplot(1, len(dists), i+1)
        # Plot the data with error bars
        # plt.errorbar(vp_mean[:, x_ind], z/1000, xerr=vp_std[:, x_ind], label='Posterior Mean', linestyle='-', color='red', capsize=4, elinewidth=0.8)
        # plt.plot(vp_median[:, x_ind], z/1000, label='Posterior Median', color='deepskyblue', linestyle='-', linewidth=2)
        plt.plot(vp_true[:, x_ind], z/1000, label='True', linestyle='-', color='k', linewidth=2)
        plt.plot(vp_fwi[:, x_ind], z/1000, label='dFWI', color='orangered', linestyle='-', linewidth=2)
        plt.plot(vp_mean[:, x_ind], z/1000, label='Mean', linestyle='-', color='blue', linewidth=2)
        plt.fill_betweenx(z/1000, vp_p10[:, x_ind], vp_p90[:, x_ind], color='deepskyblue', alpha=0.3, label='P10-P90')
        plt.ylim([z[-1]/1000, z[0]/1000])
        if vp_map is not None:
            plt.plot(vp_map[:, x_ind], z/1000, label='MAP', linestyle='-', color='green', linewidth=2)
        
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        # vp_3sigma_upper = vp_mean[:, x_ind] + 3 * vp_std[:,x_ind]
        # vp_3sigma_lower = vp_mean[:, x_ind] - 3 * vp_std[:,x_ind]

        # plot the 3-sigma region with shaded area
        # plt.fill_betweenx(z/1000, vp_3sigma_upper, vp_3sigma_lower, color='deepskyblue', alpha=0.2, label='3$\sigma$')

        plt.xlabel('Velocity (m/s)', fontsize=fontsize)
        plt.ylabel('Depth (km)', fontsize=fontsize)
        plt.title(f'Distance: {dist/1000:.2f} km', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            plt.legend(fontsize=12, loc='upper left')
            
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_samples(samples):
    ''' Plot wavefield

    Parameters
    ----------
    wavefield : 3D numpy array
        Wavefield to plot (nt, nx, ny)
        nt = number of time steps
        nx = number of grid points in x-direction
        ny = number of grid points in y-direction
    '''


    fig = plt.figure(figsize=(10, 5), dpi=100)
    ims = []
    
    caxis = abs(samples).max()
    for i in range(samples.shape[0]):
        im = plt.imshow(samples[i], vmin = -caxis, vmax = caxis, aspect = 'auto', cmap='jet', animated=False)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,repeat=True, repeat_delay=0)
    plt.close()

    return ani


def plot_dist(hmc_samples, loc1, loc2, dx, bins=40, vp_true=None, save_path=None):
    
    ix1, iz1 = np.int64(loc1[0] // dx), np.int64(loc1[1] // dx)
    ix2, iz2 = np.int64(loc2[0] // dx), np.int64(loc2[1] // dx)

    param1 = hmc_samples[:, iz1, ix1]
    param2 = hmc_samples[:, iz2, ix2]
    
    param1_name = f"m1 at ({loc1[0] / 1000:.2f} km, {loc1[1] / 1000:.2f} km)"
    param2_name = f"m2 at ({loc2[0] / 1000:.2f} km, {loc2[1] / 1000:.2f} km)"
    
    if vp_true is not None:
        ground_true = [vp_true[iz1, ix1], vp_true[iz2, ix2]]

        print(f"True value at {loc1}: {ground_true[0]:.2f} m/s")
        print(f"True value at {loc2}: {ground_true[1]:.2f} m/s")

    fontsize = 16
    plt.rcParams.update({'font.size': fontsize})

    # Create figure and gridspec
    fig = plt.figure(figsize=(8, 8))
    grid = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)

    # Top histogram (first parameter)
    ax_hist_x = fig.add_subplot(grid[0, 0:3])
    ax_hist_x.hist(param1, bins=bins, color='gray', edgecolor='white', linewidth=0.4) # skyblue
    ax_hist_x.set_xlim(param1.min(), param1.max())
    ax_hist_x.set_xticks([])
    ax_hist_x.set_yticks([])
    # plot the mean line
    ax_hist_x.axvline(np.mean(param1), color='lightcoral', linestyle='--', linewidth=1.5)

    # Right histogram (second parameter)
    ax_hist_y = fig.add_subplot(grid[1:4, 3])
    ax_hist_y.hist(param2, bins=bins, color='gray', edgecolor='white', linewidth=0.4, orientation='horizontal') # lightcoral
    ax_hist_y.set_ylim(param2.min(), param2.max())
    ax_hist_y.set_xticks([])
    ax_hist_y.set_yticks([])
    
    # plot the mean lines
    ax_hist_y.axhline(np.mean(param2), color='lightcoral', linestyle='--', linewidth=1.5)

    # Middle scatter plot with PDF
    ax_scatter = fig.add_subplot(grid[1:4, 0:3])

    # Estimate the 2D density
    xy = np.vstack([param1, param2])
    kde = gaussian_kde(xy)
    x_min, x_max = param1.min(), param1.max()
    y_min, y_max = param2.min(), param2.max()
    x, y = np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)
    xv, yv = np.meshgrid(x, y)
    density = kde(np.vstack([xv.ravel(), yv.ravel()])).reshape(xv.shape)

    # Plot the density as a color map
    density_plot = ax_scatter.contourf(xv, yv, density, levels=30, cmap='Blues', alpha=0.9)
    ax_scatter.set_xlabel(param1_name)
    ax_scatter.set_ylabel(param2_name)
    ax_scatter.grid(True)
    # ax_scatter.set_aspect('equal', 'box')
    
    # plot the mean lines
    ax_scatter.axvline(np.mean(param1), color='lightcoral', linestyle='--', linewidth=1.5)
    ax_scatter.axhline(np.mean(param2), color='lightcoral', linestyle='--', linewidth=1.5)


    # plot the true value
    if vp_true is not None:
        ax_scatter.scatter(ground_true[0], ground_true[1], color='red', marker='*', s=100)


    # Adjust the layout for better aesthetics
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()



def plot_model_dist(model_middle, model_all, selected_indices, vmin=None, vmax=None, pclip=99.0, save_path=None):
    
    
    if vmin is None or vmax is None:
        vmax = np.percentile(model_middle, pclip)
        vmin = -vmax
        
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=(12, 5), dpi=150)
    gs = gridspec.GridSpec(4, 4)

    # Assign subplot locations for histograms (non-central cells)
    hist_positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 3),
        (2, 0), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3)
    ]

    # Create and plot histograms
    for idx, ((i, j), (row, col)) in enumerate(zip(selected_indices, hist_positions)):
        ax = fig.add_subplot(gs[row, col])
        samples = model_all[:, i, j]
        ax.hist(samples, bins=30, color='dodgerblue', alpha=0.7)
        ax.text(0.95, 0.95, f'({i}, {j})', transform=ax.transAxes, 
                fontsize=8, ha='right', va='top', color='black')
        ax.tick_params(labelsize=6)
        ax.set_xlim([0, 1])
        # ax.set_ylim([0, 2000])

    # Central posterior mean plot at [1:3, 1:3]
    ax_center = fig.add_subplot(gs[1:3, 1:3])
    im = ax_center.imshow(model_middle, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    cax = inset_axes(ax_center, width="2%", height="50%", loc='lower right', borderpad=4)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=6)

    # Mark the 12 points on the center plot
    for idx, (i, j) in enumerate(selected_indices):
        ax_center.plot(j, i, 'ko', markersize=4)
        ax_center.text(j+1, i+1, str(idx+1), color='k', fontsize=6)

    plt.tight_layout()
    plt.show()
    
    
    
    
    


def plot_2d(x, z, d, centers=None, same_clip=True, pclip=98, aspect='auto', 
            title=None, vmin=None, vmax=None, cmap='jet', label=False, 
            label_color='w', save_path=None, dpi=150, xlim=None, ylim=None, layout='horizontal',
            figsize = (10, 5)):
    # Check if 'd' is a list of arrays
    if isinstance(d, list):
        plt.figure(figsize=figsize, dpi=dpi)
        
        # Define initial clip value based on the first element in the list
        clip_plot_max = np.percentile(np.abs(d[0]), pclip)
        clip_plot_min = -clip_plot_max
        if vmin is not None:
            clip_plot_min = vmin
        if vmax is not None:
            clip_plot_max = vmax
        
        for i, dd in enumerate(d):
            # Update clip value if different clips are used for each plot
            if not same_clip:
                clip_plot = np.percentile(np.abs(dd), pclip)

            # Plot each item in the list 'd'
            if layout == 'horizontal':
                ax = plt.subplot(1, len(d), i + 1)
            elif layout == 'vertical':
                ax = plt.subplot(len(d), 1, i + 1)
            else:
                raise ValueError("not supported")
            img = ax.imshow(dd, extent=[x[0], x[-1], z[-1], z[0]], aspect=aspect, cmap=cmap,
                            vmin=clip_plot_min, vmax=clip_plot_max)
            ax.grid(linestyle='--', alpha=0.5)
            plt.colorbar(img, ax=ax, orientation='horizontal')
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Depth (m)")
            if title is not None:
                ax.set_title(title[i])

    else:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)
        # Calculate clip value for a single 2D array
        clip_plot_max = np.percentile(np.abs(d), pclip)
        clip_plot_min = -clip_plot_max
        if vmin is not None:
            clip_plot_min = vmin
        if vmax is not None:
            clip_plot_max = vmax
            
        img = plt.imshow(d, extent=[x[0], x[-1], z[-1], z[0]], aspect=aspect, cmap=cmap,
                   vmin=clip_plot_min, vmax=clip_plot_max)
        
        # Plot centers if provided
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], marker='.', color='r')
        
        if label:
            # plot the label for each point
            for i, txt in enumerate(centers):
                ax.annotate(f'{i}', (txt[0], txt[1]), color=label_color, fontsize=8)
        
        if xlim is not None:
            plt.xlim(xlim)
        
        if ylim is not None:
            plt.ylim(ylim)
        plt.grid(linestyle='--', alpha=0.5)
        plt.colorbar(img, ax=ax, orientation='horizontal')
        plt.xlabel("Distance (m)")
        plt.ylabel("Depth (m)")

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

