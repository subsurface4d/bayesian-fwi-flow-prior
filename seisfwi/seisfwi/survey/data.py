import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List, Union
from seisfwi.survey import Survey



class SeismicData():
    """ Seismic data class in shot gather format
    """

    def __init__(self, survey: Survey):
        """ Init the shot gather data class

        Parameters:
        ----------
        Survey: Survey
            Survey class
        """

        # get the survey information
        self.src_num = survey.source.num
        self.rec_num = survey.receiver.num
        self.src_loc = survey.source.get_loc()
        self.rec_loc = survey.receiver.get_loc_dict()
        self.rec_type = survey.receiver.get_type()
        self.src_type = survey.source.get_type()
        self.nt = survey.receiver.nt
        self.dt = survey.receiver.dt
        self.t = np.arange(self.nt) * self.dt  # start from 0 ms

        self.first_break = None  # in seconds

        # initialize the data dictionary as None
        self.data = None

    def __repr__(self):
        """ Print the survey information
        """

        info = f"Seismic Data:\n"
        info += f"  Source number : {self.src_num}\n"
        info += f"  Receiver number : {self.rec_num}\n"
        info += f"  Time samples : {self.nt} samples at {self.dt * 1000:.2f} ms\n"
        if self.data is not None:
            comp = list(self.data.keys())[0]
            data_min = self.data[comp].min().item() 
            data_max = self.data[comp].max().item()
            device = self.data[comp].device
            dtype = self.data[comp].dtype                
   
            info += f"  Data range: {data_min:.2e} - {data_max:.2e}\n"
            info += f"  Data device: {device} with dtype {dtype}\n"

        return info

    @classmethod
    def load_from_segy_files(self):
        """ Load the shot gather data from segy files
        """

        raise NotImplementedError('This function is not implemented yet')

    @classmethod
    def load_from_sep_files(self):
        """ Load the shot gather data from sep files
        """

        raise NotImplementedError('This function is not implemented yet')

    def record_data(self, data: dict):
        """ Add the shot gather data to the class

        Parameters:
        ----------
        data: dict
            shot gather data in dictionary format
        """

        self.data = data

    def save(self, path: str):
        """ Save the shot gather data

        Parameters:
        ----------
        path: str
            save path
        """

        data = {comp: t.detach() for comp, t in self.data.items()}

        data_save = {'data': data,
                    'src_loc': self.src_loc,
                    'rec_loc': self.rec_loc,
                    'src_num': self.src_num,
                    'rec_num': self.rec_num,
                    'rec_type': self.rec_type,
                    'src_type': self.src_type,
                    't': self.t,
                    'nt': self.nt,
                    'dt': self.dt}
        
        np.savez(path, **data_save) 


    @classmethod
    def load(cls, path: str):
        """ Load the shot gather data

        Parameters:
        ----------
        path: str
            load path
        """

        data = np.load(path, allow_pickle=True)

        # create the SeismicData object
        seismic_data = cls.__new__(cls)

        # load the data
        seismic_data.data = data['data'].item()
        seismic_data.src_loc = data['src_loc']
        seismic_data.rec_loc = data['rec_loc'].item()
        seismic_data.src_num = data['src_num']
        seismic_data.rec_num = data['rec_num']
        seismic_data.rec_type = data['rec_type']
        seismic_data.src_type = data['src_type']
        seismic_data.t = data['t']
        seismic_data.nt = data['nt']
        seismic_data.dt = data['dt']

        return seismic_data


    def check_same(self, other: 'SeismicData'):
        """ Check if the shot gather data is the same

        Parameters:
        ----------
        other: SeismicData
            another SeismicData object to compare with

        Raises:
        ----------
        TypeError: if the input is not a SeismicData object
        ValueError: if the shot gather data is not the same
        ... 
        """
        
        if not isinstance(other, SeismicData):
            raise TypeError('Input must be a SeismicData object.')

        # check if the shot gather data is the same
        if not np.all(self.src_loc == other.src_loc):
            raise ValueError('The source locations are not the same')
        if not self.rec_loc.keys() == other.rec_loc.keys():
            raise ValueError('The receiver components are not the same')
        for comp in self.rec_loc.keys():
            if not np.all(np.array(self.rec_loc[comp]) == np.array((other.rec_loc[comp]))):
                raise ValueError(f'The receiver locations for component {comp} are not the same')
        if not self.nt == other.nt:
            raise ValueError('The time samples are not the same')
        if not self.dt == other.dt:
            raise ValueError('The time sampling interval are not the same')
        
        
    def get_data(self, shotid: Optional[int] = 0, comp: Optional[str] = 'vz'):
        """ Get the shot gather data

        Parameters:
        ----------
        shotid: int
            shot id
        comp: str
            component to get
        
        Returns:
        ----------
        sg_data: np.ndarray
            shot gather data in NumPy array format
        """

        if self.data is None:
            raise ValueError('The shot gather data is not available, forward modeling is needed')
        
        if comp not in self.rec_type:
            raise ValueError(f'The component {comp} is not available, available components are {set(self.rec_type)}')  
        
        if shotid > self.src_num - 1 :
            raise ValueError(f'The shot id {shotid} is not available, available shot ids are 0 - {self.src_num-1}')

        # extract the data and convert to NumPy array
        sg_data = self.data[comp][shotid]

        if not isinstance(sg_data, np.ndarray):
            try:
                sg_data = sg_data.cpu().detach().numpy()
            except AttributeError:
                # Handle the case where data cannot be converted to a NumPy array
                pass
        
        return sg_data


    def plot(self, shotid: Optional[int] = 0, comp: Optional[Union[str, List[str]]] = 'vz', 
             pclip: Optional[float] = 99.9, cmap: Optional[str] = 'gray', 
             aspect: Optional[Union[str, float]] = 'auto', 
             time_range: Optional[List[float]] = None,
             figsize: Optional[tuple] = (4,6), 
             show_colorbar: Optional[bool] = False,
             show_first_break: Optional[bool] = False,
             save_path: Optional[str] = None):
        
        """ Plot the shot gather data
    
        Parameters:
        ----------
        shotid: int
            shot id
        comp: str or list of str
            component to plot
        clip: float
            clip value
        cmap: str
            colormap
        aspect: str or float
            aspect ratio
        figsize: tuple
            figure size
        show_colorbar: bool
            show colorbar or not
        show_first_break: bool
            show first break or not
        save_path: str
            figure save path, if None, do not save
        """

        if isinstance(comp, str):
            comp = [comp]
        
        for c in comp:
            if c not in set(self.rec_type):
                print(f'The component {comp} is not available, available components are {set(self.rec_type)}') 
                # return None
        comp = list(set(self.rec_type))
        
        figsize = (figsize[0] * len(comp), figsize[1])

        fig = plt.figure(figsize=figsize)
        
        for i, c in enumerate(comp):

            # extract the data to plot
            data = self.get_data(shotid, c)

            channel = np.arange(data.shape[0])

            vmax = np.percentile(data, pclip)

            # set up the figure
            parms = {'cmap':cmap, 
                    'aspect':aspect, 
                    'vmin':-vmax, 
                    'vmax':vmax,
                    'extent': [channel[0], channel[-1], self.t[-1], self.t[0]]}

            ax = fig.add_subplot(1,len(comp),i+1)
            ax.imshow(data.T, **parms)
            ax.set_xlabel('Channel#')
            if i == 0: 
                ax.set_ylabel('Time (s)')
            ax.set_title(f' Shot #{shotid} ({c.upper()})')
            ax.grid(True, axis='y', alpha=0.5)
            if time_range:
                ax.set_ylim(time_range[1], time_range[0])
            if show_colorbar:
                fig.colorbar(ax.images[0], ax=ax, orientation='vertical')

            if show_first_break:
                if self.first_break is None:
                    print('The first break is not picked, please run the first break picking algorithm first')
                else:
                    ax.plot(channel, self.first_break[c][shotid], 'r-', linewidth=1.5)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def plot_compare(self, shotdata: 'SeismicData',
                        shotid: Optional[int] = 0, 
                        comp: Optional[str] = 'vz', 
                        pclip: Optional[float] = 99.9, 
                        ratio: Optional[float] = 1.0,
                        cmap: Optional[str] = 'gray', 
                        depth: Optional[float] = None,
                        normalize: Optional[bool] = False,
                        time_range: Optional[List[float]] = None,
                        aspect: Optional[Union[str, float]] = 'auto', 
                        figsize: Optional[tuple] = (15,8), 
                        fontsize: Optional[int] = 12,
                        dpi: Optional[int] = 100,
                        show_colorbar: Optional[bool] = False,
                        title: Optional[str] = ['Data 1', 'Data 2', 'Difference'],
                        save_path: Optional[str] = None):

        """ Compare two shot gather data based on the specified method

        Parameters:
        ----------
        shotdata: SeismicData
            another SeismicData object to compare with
        shotid: int
            shot id
        comp: str
            component to plot
        clip: float
            clip value
        ratio: float
            ratio of difference data
        cmap: str
            colormap
        aspect: str
            aspect ratio
        figsize: tuple
            figure size
        show_colorbar: bool
            show colorbar or not
        save_path: str
            figure save path, if None, do not save
        """

        # check if the shot gather data is the same
        self.check_same(shotdata)
        
        for c in comp:
            if c not in set(self.rec_type):
                print(f'The component {comp} is not available, available components are {set(self.rec_type)}') 
                # return None
        comp = self.rec_type[0]
        
        data1 = self.get_data(shotid, comp)
        data2 = shotdata.get_data(shotid, comp)

        if normalize:
            data1_max = np.amax(np.abs(data1), axis=-1, keepdims=True)
            data1_max = np.where(data1_max == 0, 1.0, data1_max)
            data1 /= data1_max

            data2_max = np.amax(np.abs(data2), axis=-1, keepdims=True)
            data2_max = np.where(data2_max == 0, 1.0, data2_max)
            data2 /= data2_max

        # implement the comparison method for other methods
        data_diff = data1 - data2

        if depth is None:
            channel = np.arange(data1.shape[0])
            xlabel = 'Channel#'
        else:
            channel = depth
            xlabel = 'Depth (m)'
            
        vmax = np.percentile(data1, pclip)

        vmaxs = [vmax, vmax, vmax / ratio]
        title = [title[0], title[1], title[2] + f' (x{ratio})']
 
        fig = plt.figure(figsize=figsize, dpi=dpi)

        for i, data in enumerate([data1, data2, data_diff]):

            # set up the figure
            parms = {
                'cmap': cmap,
                'aspect': aspect,
                'vmin': -vmaxs[i],
                'vmax': vmaxs[i],
                'extent': [channel[0], channel[-1], self.t[-1], self.t[0]],
            }

            ax = fig.add_subplot(1, 3, i + 1)
            im = ax.imshow(data.T, **parms)

            ax.set_xlabel(xlabel, fontsize=fontsize)
            if i == 0:
                ax.set_ylabel("Time (s)", fontsize=fontsize)

            ax.set_title(title[i], fontsize=fontsize)
            ax.grid(True, axis="y", alpha=0.5)

            # set tick font size
            ax.tick_params(axis="both", which="major", labelsize=fontsize)

            if time_range is not None:
                ax.set_ylim(time_range[1], time_range[0])

            if show_colorbar:
                cbar = fig.colorbar(im, ax=ax, orientation="vertical")
                cbar.ax.tick_params(labelsize=fontsize)  # colorbar tick labels

        plt.tight_layout()


        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    # def plot_dispersion(self, shotid: Optional[int] = 0,
    #                     dx= None,
    #                     comp: Optional[str] = 'vz',
    #                     vmin: Optional[float] = 200.0,
    #                     vmax: Optional[float] = 4000.0,
    #                     fmin: Optional[float] = 0.0,
    #                     fmax: Optional[float] = 50.0,
    #                     dv: Optional[float] = 10.0,
    #                     method: Optional[str] = 'fk',
    #                     normalize: Optional[bool] = False,
    #                     clip: Optional[float] = 99.9,
    #                     cmap: Optional[str] = 'gray',
    #                     aspect: Optional[Union[str, float]] = 'auto',
    #                     figsize: Optional[tuple] = (8,6),
    #                     show_colorbar: Optional[bool] = False,
    #                     save_path: Optional[str] = None,
    #                     return_data: Optional[bool] = False):

    #     """ Plot the dispersion curve of the shot gather data, based on the FK analysis

    #     Parameters:
    #     ----------
    #     shotid: int
    #         shot id
    #     comp: str
    #         component to plot
    #     vmin: float
    #         minimum phase velocity
    #     vmax: float
    #         maximum phase velocity
    #     fmin: float
    #         minimum frequency
    #     fmax: float
    #         maximum frequency
    #     normalize: bool
    #         normalize the data or not along the frequency axis
    #     clip: float
    #         clip value
    #     cmap: str
    #         colormap
    #     aspect: str or float
    #         aspect ratio
    #     figsize: tuple
    #         figure size
    #     show_colorbar: bool
    #         show colorbar or not
    #     save_path: str
    #         figure save path, if None, do not save
    #     """


    #     # extract the data to plot
    #     data = self.get_data(shotid, comp)

    #     if dx is None:
    #         raise ValueError('The spatial sampling interval dx is not set, please set it manually')
            
    #     # set the spatial sampling interval
    #     # rec1 = self.rec_loc[comp][0]
    #     # rec2 = self.rec_loc[comp][1]
    #     # dx = np.sqrt((rec2[0] - rec1[0])**2 + (rec2[1] - rec1[1])**2)        

    #     # set the temporal sampling interval
    #     dt = self.dt

    #     # print the sampling interval
    #     # print(f' dx = {dx:.2f} m')
    #     # print(f' dt = {dt * 1000:.2f} ms')


    #     if method == 'fk':

    #         # compute the dispersion curve
    #         f, k, d_fk_r = fk(data, dt, dx, )
    #         f, c, d_fs = fc(d_fk_r, f, k, c_min=vmin, c_max=vmax)

    #         # select the data to plot
    #         ifmin = np.where(np.abs(f-fmin)==np.min(np.abs(f-fmin)))[0][0]
    #         ifmax = np.where(np.abs(f-fmax)==np.min(np.abs(f-fmax)))[0][0]
    #         icmax = np.where(np.abs(c-vmin)==np.min(np.abs(c-vmin)))[0][0]
    #         icmin = np.where(np.abs(c-vmax)==np.min(np.abs(c-vmax)))[0][0]
            
    #         f_plot = f[ifmin:ifmax]
    #         c_plot = c[icmin:icmax]
    #         d_fs_plot = d_fs[icmin:icmax, ifmin:ifmax]

    #         if normalize:
    #             d_fs_plot = d_fs_plot / np.max(d_fs_plot, axis=0)
    #             # d_fs[np.isnan(d_fs)] = 0.0

    #         vmax = np.percentile(d_fs_plot, clip)

    #         # plot the dispersion curve
    #         fig = plt.figure(figsize=figsize)

    #         ax = fig.add_subplot(1,1,1)
    #         plt.pcolor(f_plot, c_plot, d_fs_plot, cmap=cmap, clim=(0.0, vmax), shading='auto')
    #         ax.set_xlabel('Frequency (Hz)')
    #         ax.set_ylabel('Phase Velocity (m/s)')
    #         ax.set_title(f' Dispersion Curve of Shot #{shotid} ({comp.upper()})')
    #         ax.grid(True, axis='y', alpha=0.5)
    #         plt.gca().set_aspect(aspect, adjustable='box')

    #         if show_colorbar:
    #             plt.colorbar(orientation='horizontal')
            
    #         plt.tight_layout()
    #         if save_path is not None:
    #             plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
    #         plt.show()

    #     elif method == 'phase_shift':
            
    #         rec_num, nt = data.shape
    #         rec_x = np.arange(rec_num) * dx + self.rec_loc[comp][0][0]
    #         src_x = self.src_loc[shotid][0]
    #         offset = rec_x - src_x

    #         # Find valid traces (non-zero rows)
    #         traces_use = np.where(np.any(data != 0, axis=1))[0]

    #         # Filter the data and offset accordingly
    #         data_use = data[traces_use, :]
    #         offset_use = offset[traces_use]

    #         # Separate positive and negative offsets
    #         pos_idx = offset_use > 0
    #         neg_idx = offset_use < 0

    #         # Initialize
    #         fv_data_pos = 0.0
    #         fv_data_neg = 0.0

    #         # Phase shift for positive offsets
    #         if np.sum(pos_idx) > 3:
    #             fv_data_pos, f_axis, v_axis = phase_shift(
    #                 data_use[pos_idx, :], dt, offset_use[pos_idx], 
    #                 vmin, vmax, dv, fmin, fmax, smooth_factor=5
    #             )

    #         # # Phase shift for negative offsets
    #         # if np.sum(neg_idx) > 3:
    #         #     fv_data_neg, f_axis, v_axis = phase_shift(
    #         #         data_use[neg_idx, :], dt, -offset_use[neg_idx], 
    #         #         vmin, vmax, dv, fmin, fmax, smooth_factor=5
    #         #     )

    #         # Negative offsets (flip data and offset)
    #         if np.sum(neg_idx) > 3:
    #             data_neg_flipped   =  np.flipud(data_use[neg_idx, :])      # Flip trace order
    #             offset_neg_flipped = -np.flip(offset_use[neg_idx])               # Make offsets positive and flip order
                
    #             fv_data_neg, f_axis, v_axis = phase_shift(
    #                 data_neg_flipped, dt, offset_neg_flipped,
    #                 vmin, vmax, dv, fmin, fmax, smooth_factor=5
    #             )

    #         # Combine both
    #         fv_data = fv_data_pos + fv_data_neg

    #         if normalize:
    #             fv_data = fv_data / np.max(fv_data, axis=0)
            
    #         plt.figure(figsize=figsize)
    #         plt.subplot(1, 1, 1)
    #         plt.imshow(fv_data, aspect="auto",  extent=(f_axis[0], f_axis[-1], v_axis[-1], v_axis[0]), cmap='jet', 
    #                 interpolation='bilinear', vmin=0, vmax=1)
    #         plt.xlim(f_axis[0], f_axis[-1])
    #         plt.ylim(v_axis[0], v_axis[-1])
    #         plt.xlabel('Frequency (Hz)')
    #         plt.ylabel('Phase Velocity (m/s)')
    #         plt.title(f'Dispersion Curve')
    #         plt.grid(True, alpha=0.5)
    #         plt.gca().set_aspect(aspect, adjustable='box')
    #         if show_colorbar:
    #             plt.colorbar(orientation='horizontal')
        
    #         # plt.subplot(1, 2, 2)
    #         # plt.imshow(fv_data_pos, aspect="auto",  extent=(f_axis[0], f_axis[-1], v_axis[-1], v_axis[0]), cmap='jet', 
    #         #         interpolation='bilinear', vmin=0, vmax=1)
    #         # plt.xlim(f_axis[0], f_axis[-1])
    #         # plt.ylim(v_axis[0], v_axis[-1])
    #         # plt.xlabel('Frequency (Hz)')
    #         # plt.ylabel('Phase Velocity (m/s)')
    #         # plt.title(f'Positive Offset Dispersion Curve')
    #         # plt.grid(True, alpha=0.5)
    #         # plt.gca().set_aspect(aspect, adjustable='box')            
    #         # if show_colorbar:
    #         #     plt.colorbar(orientation='horizontal')
        
    #         plt.tight_layout()
    #         if save_path is not None:
    #             plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
    #         plt.show()

    #     if return_data:
    #         return f_plot, c_plot, d_fs_plot

    def plot_trace(self, shotid: Optional[int] = 0, 
             traceid: Optional[int] = 0,
             comp: Optional[Union[str, List[str]]] = 'vz', 
             figsize: Optional[tuple] = (8,3), 
             save_path: Optional[str] = None):
        """ Plot the trace

        Parameters:
        ----------
        shotid: int
            shot id
        traceid: int
            trace id
        comp: str or list of str
            component to plot
        figsize: tuple
            figure size
        save_path: str
            figure save path, if None, do not save
        """

        if isinstance(comp, str):
            comp = [comp]

        figsize = (figsize[0] * len(comp), figsize[1])

        fig = plt.figure(figsize=figsize)

        for i, c in enumerate(comp):
                
                # extract the data to plot
                data = self.get_data(shotid, c)
    
                ax = fig.add_subplot(1,len(comp),i+1)
                ax.plot(self.t, data[traceid], 'k-')
                ax.set_ylabel('Amplitude')
                if i == 0: 
                    ax.set_xlabel('Time (s)')
                ax.set_title(f' Shot #{shotid} ({c.upper()})')
                ax.grid(True, axis='y', alpha=0.5)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def plot_compare_trace(self, shotdata: 'SeismicData',
                            shotid: Optional[int] = 0,
                            traceid: Optional[int] = 0,
                            comp: Optional[str] = 'vz',
                            linewidth: Optional[float] = 1.5,
                            linestyle: Optional[List[str]] = ['k-', 'r-'],
                            figsize: Optional[tuple] = (8,3),
                            title: Optional[str] = ['Data 1', 'Data 2'],
                            normlize: Optional[bool] = False,
                            save_path: Optional[str] = None):
        
        """ Compare two shot gather data based on the specified method

        Parameters:
        ----------
        shotdata: SeismicData
            another SeismicData object to compare with
        shotid: int
            shot id
        traceid: int
            trace id
        comp: str
            component to plot
        figsize: tuple
            figure size
        save_path: str
            figure save path, if None, do not save
        """

        # check if the shot gather data is the same
        self.check_same(shotdata)

        data1 = self.get_data(shotid, comp)
        data2 = shotdata.get_data(shotid, comp)
        
        trace1 = data1[traceid]
        trace2 = data2[traceid]

        if normlize:
            trace1 = trace1 / (np.max(abs(trace1)) + 1e-10)
            trace2 = trace2 / (np.max(abs(trace2)) + 1e-10)

        # implement the comparison method for other methods
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1,1,1)
        ax.plot(self.t, trace1, linestyle[0], linewidth=linewidth, label=title[0])
        ax.plot(self.t, trace2, linestyle[1], linewidth=linewidth, label=title[1])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title[0])
        ax.grid(True, axis='y', alpha=0.5)
        ax.legend()

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


    # def filter_bandpass(self, lowcut = 2, highcut = 10, order = 4):
    #     """ Filter the shot gather data in frequency domain

    #         To maintain consistency, this part uses the filter in numpy/scipy. However,
    #         The data copy here is not efficient, consider using the inplace operation
    #         later
    #     """

    #     dt = self.dt
    #     dtype = self.data[self.rec_type[0]].dtype
    #     device = self.data[self.rec_type[0]].device

    #     for comp in self.data.keys():
    #         for isrc in range(self.src_num):
    #             data = self.get_data(isrc, comp)
    #             data_bp = bandpass_filter(data, dt, lowcut, highcut, order=order)

    #             # copy the filtered data back
    #             self.data[comp][isrc] = torch.tensor(data_bp.copy(), dtype=dtype, device=device)

    #     print(f'Bandpass filter the data: {lowcut} - {highcut} Hz')


    # def pick(self, threshold = 0.01):
    #     """ Pick the first arrival of the shot gather data

    #     Consider using the ML package to pick the first arrival here

    #     Parameters:
    #     ----------
    #     threshold: float
    #         threshold for first break picking   
    #     """
        
    #     self.first_break = {}

    #     for comp in set(self.rec_type):

    #         src_num, rec_num = self.data[comp].shape[:2]
    #         first_break = np.zeros((src_num, rec_num))

    #         for i in range(src_num):
    #             for j in range(rec_num):

    #                 # Extract the tensor for the current indices
    #                 current_trace = self.data[comp][i, j]

    #                 # Compute the threshold based on the maximum value of the current tensor
    #                 current_threshold = current_trace.abs().max() * threshold

    #                 # Find the index of the first element where the absolute value exceeds the threshold
    #                 pick_index = (current_trace.abs() >= current_threshold).float().argmax()

    #                 # Store the result in the 'first_break' array
    #                 first_break[i, j] = pick_index.item()

    #         # Convert the first break to seconds
    #         self.first_break[comp] = first_break * self.dt

    #     print('First break picking is done')


    # def mute_offset(self, offset: Optional[float] = 1000.0, near_offset: Optional[bool] = True):
    #     """ Mute the shot gather data based on offset

    #     Parameters:
    #     ----------
    #     offset: float
    #         offset to mute.
    #         For near offset, the offset is smaller than the threshold
    #         For far offset, the offset is larger than the threshold
    #     near_offset: bool
    #         mute near offsets or far offsets
    #     """

    #     for comp in set(self.rec_type):
    #         src_num, rec_num = self.data[comp].shape[:2]

    #         for i in range(src_num):
    #             for j in range(rec_num):

    #                 # Compute the offset of the current trace
    #                 current_offset = np.sqrt((self.src_loc[i][0] - self.rec_loc[comp][j][0])**2 + 
    #                                          (self.src_loc[i][1] - self.rec_loc[comp][j][1])**2)

    #                 # Mute the current trace if the offset is larger than the threshold
    #                 if near_offset:
    #                     if current_offset < offset:
    #                         self.data[comp][i, j] *= 0.0
    #                 else:
    #                     if current_offset > offset:
    #                         self.data[comp][i, j] *= 0.0

    #     if near_offset:
    #         print(f'Mute near offsets with offset {offset:.2f} m')
    #     else:
    #         print(f'Mute far offsets with offset {offset:.2f} m')


    
    # def mute_arrival(self, window_size: Optional[float] = 0.2, late_arrival: Optional[bool] = True):
    #     """ Mute the shot gather data based on time window

    #     Parameters:
    #     ----------
    #     window_size: float
    #         time window size in seconds
    #     late_arrival: bool
    #         mute late arrival or not
    #     """
        

    #     if self.first_break is None:
    #         print('The first break is not picked, please run the first break picking algorithm first')

    #         return None

    #     device = self.data[self.rec_type[0]].device

    #     # taper with default length of 20 samples
    #     length = 20
    #     taper = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, length))
    #     taper = taper.to(device)

    #     # mute the data
    #     for comp in set(self.rec_type):
    #         src_num, rec_num = self.data[comp].shape[:2]

    #         for i in range(src_num):
    #             for j in range(rec_num):

    #                 # Extract the tensor for the current indices
    #                 itmin = int((self.first_break[comp][i, j] + window_size) / self.dt)
    #                 itmax = itmin + length

    #                 mask = torch.ones(self.nt, device=device)

    #                 if late_arrival:
    #                     self.data[comp][i, j] *= (1.0 - custom_mask(itmin, itmax, self.nt, length, taper, mask))
    #                 else:
    #                     self.data[comp][i, j] *= custom_mask(itmin, itmax, self.nt, length, taper, mask)
    
    #     if late_arrival:
    #         print(f'Mute late arrivals with window size {window_size * 1000:.2f} ms after first break')
    #     else:
    #         print(f'Mute early arrivals with window size {window_size * 1000:.2f} ms after first break')




    # def select_time_window(self, tmin: Optional[float] = 0.0, tmax: Optional[float] = None, nt: Optional[int] = None):
    #     """ Select the time window of the shot gather data

    #     Parameters:
    #     ----------
    #     tmin: float
    #         minimum time
    #     tmax: float 
    #         maximum time 
    #     nt: int
    #         number of time samples
    #     """

    #     if tmax is None:
    #         tmax = self.t[-1]

    #     if nt is not None:
    #         tmax = self.t[nt-1]

    #     # select the time window
    #     index = np.arange(self.nt)
    #     index = index[(self.t >= tmin) & (self.t <= tmax)]

    #     self.t = self.t[index]
    #     self.nt = len(self.t)

    #     print(f'Select time window: {tmin * 1000:.2f} ms - {tmax * 1000:.2f} ms')
    #     print(f'New time samples: {self.nt} samples at {self.dt * 1000:.2f} ms')

    #     # select the data
    #     for comp in set(self.rec_type):
    #         self.data[comp] = self.data[comp][:, :, index]

    # def resample(self, dt: float):
    #     """ Resample the shot gather data

    #     Parameters:
    #     ----------
    #     dt: float
    #         new time sampling interval
    #     """

    #     if dt == self.dt:
    #         print(f'The data is already sampled at {dt * 1000:.2f} ms')
    #         return
        
    #     # resample the time samples
    #     raise NotImplementedError('This function is not implemented yet')
      

    #     print(f'Resample the data: {self.dt * 1000:.2f} ms -> {dt * 1000:.2f} ms')
    #     print(f'New time samples: {self.nt} samples at {dt * 1000:.2f} ms')
