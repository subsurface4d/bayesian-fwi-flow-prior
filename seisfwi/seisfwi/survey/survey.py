from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import torch
import logging

import seisfwi.defaults as defaults
from seisfwi.survey import Receiver
from seisfwi.survey import Source

# Get your module logger
logger = logging.getLogger(__name__)


class Survey(object):
    """Survey class describes the seismic acquisition geometry (2D). I assume 
    that all sources share the same receivers, time samples, and time interval.

    Parameters
    ----------
    source : Source 
        Source object
    receiver : Receiver
        Receiver object
    device : str, optional
        Device for computation: cpu or gpu, by default 'cpu'
    cpu_num : int, optional
        Maximum number of CPU cores, if cpu, by default 1
    gpu_num : int, optional
        Maximum number of GPU cards, if cuda, by default 1
    reciprocity : bool, optional
        Apply reciprocity to the survey or not, by default False
    simultaneous : bool, optional
        Apply simultaneous source acquisition to the survey or not, by default False
    interpolation : bool, optional
        Apply interpolation to the source and receiver or not, by default False
        This method is based on the Kaiser-Sinc interpolation method (Hicks, 2002)

    Notes
    -----
        1. The receivers are assumed to be the same for all shots, i.e., the
              receiver locations are not shot-dependent. See the notes in Receiver
                class for more details.
        2. 2-D simulation for now, i.e., (x, z) coordinates.
        3. The parallelization is in a shot-by-shot fashion, not domain decomposition.
           In fact, multiple shots can be run simultaneously on one GPU, as long 
           as the memory is enough for cached the wavefields for computing the
           gradient.
        4. Interpolation is time-consuming, but it is necessary for DAS acquisition.

    TODO
    ----
        1. Speed up the interpolation code
    """

    def __init__(
        self,
        source: Source,
        receiver: Receiver,
        cpu_num: Optional[int] = 1,
        gpu_num: Optional[int] = 1,
        reciprocity: Optional[bool] = False,
        simultaneous: Optional[bool] = False,
        interpolation: Optional[bool] = False,
    ) -> None:
        
        # Survey geometry
        self.source = source
        self.receiver = receiver

        # Compute resource config
        self.device = defaults.device
        self.cpu_num = cpu_num
        self.gpu_num = gpu_num

        # Options
        self.reciprocity = reciprocity
        self.simultaneous = simultaneous
        self.interpolation = interpolation

        # Check validity
        self._check()


    def __repr__(self):
        """ Reimplement the repr function for printing the survey information
        """

        info = f"Survey Information:\n"
        info += f"  Device   : {self.device}\n"
        info += f"  CPU num  : {self.cpu_num}\n" if defaults.device.type == "cpu" else ""
        info += f"  GPU num  : {self.gpu_num}\n" if defaults.device.type == "cuda" else ""
        info += f"  Apply reciprocity: {self.reciprocity}\n"
        info += f"  Simultaneous source: {self.simultaneous}\n"
        info += f"  Apply interpolation: {self.interpolation}\n"
        info += "\n"
        info += repr(self.source)
        info += "\n"
        info += repr(self.receiver)
        info += "\n"
        if self.receiver.cable is not None:
            info += repr(self.receiver.cable)

        return info

    def _check(self):
        """
        Validate the survey configuration.
        """
        assert self.source.nt == self.receiver.nt, "Source and receiver must have same nt."
        assert self.source.dt == self.receiver.dt, "Source and receiver must have same dt."
        assert self.cpu_num > 0 and self.gpu_num > 0, "CPU and GPU counts must be positive."

        # Check the number of available GPUs
        gpu_num_avail = torch.cuda.device_count()

        if self.device.type == "cuda":
            if gpu_num_avail < 1:
                logger.warning("No GPUs detected — switching to CPU.")
                self.device = torch.device("cpu")
                self.gpu_num = 0

            elif self.gpu_num > gpu_num_avail:
                logger.warning(
                    f"Requested {self.gpu_num} GPUs but only {gpu_num_avail} available. Using {gpu_num_avail}."
                )
                self.gpu_num = gpu_num_avail

            if self.gpu_num > self.source.num:
                logger.warning(
                    f"Requested {self.gpu_num} GPUs but only {self.source.num} shots. Using {self.source.num}."
                )
                self.gpu_num = self.source.num

        if hasattr(self.receiver, "cable") and self.receiver.cable is not None:
            if not self.interpolation:
                logger.info(
                    "For DAS receivers, survey interpolation is strongly recommended."
                )

    def initialize(self, dx: float, dz: float) -> None:
        """Initialize the survey
        
        Parameters
        ----------
        dx : float
            Spatial interval in x-direction (m)
        dz : float
            Spatial interval in z-direction (m)

        Notes
        -----
            This function should be called before running the simulation.

        """

        assert dx == dz, "dx and dz should be the same for this survey"

        self.dx = dx
        self.dz = dz

        # set the cable operator if the cable exists
        if self.receiver.cable is not None:
            self.cable_operator = self.receiver.cable.get_operator(device=self.device)  

        # apply reciprocity if needed
        if self.reciprocity:
            self.__apply_reciprocity()

        # set the source and receiver
        self.__set_source()
        self.__set_receiver()


    def __apply_reciprocity(self) -> None:
        """Apply reciprocity to the survey, 
        
        Notes
        -----
        This assumes the same source wavelet for all sources.
        """

        print("Applying reciprocity to the survey ...")
        print("  assuming the same source wavelet for all sources!")

        # ---------------------------------------------------------------
        # apply reciprocity to the receiver
        # ---------------------------------------------------------------

        # check the source type for applying reciprocity
        src_type = self.source.get_type(unique=True)
        if len(src_type) > 1:
            raise ValueError(
                "Reciprocity Error: source type should be the same for all sources"
            )

        src_type = src_type[0]
        if src_type != "pr":
            raise ValueError("Reciprocity Error: source type should be pressure source")

        # set up new receiver
        receiver = Receiver(nt=self.receiver.nt, dt=self.receiver.dt)
        src_loc = self.source.get_loc(type=src_type)

        for isrc in range(src_loc.shape[0]):
            receiver.add_receiver(src_loc[isrc] + np.array([0.0,      0.0]), "vx")
            receiver.add_receiver(src_loc[isrc] + np.array([self.dx,  0.0]), "vx")
            receiver.add_receiver(src_loc[isrc] + np.array([0.0, -self.dz]), "vz")
            receiver.add_receiver(src_loc[isrc] + np.array([0.0,      0.0]), "vz")

        # ---------------------------------------------------------------
        # apply reciprocity to the source
        # Note: the source wavelet should be changed below for different source types
        # ---------------------------------------------------------------

        # create new source and receiver
        source = Source(nt=self.source.nt, dt=self.source.dt, f0=self.source.f0)

        # set source wavelet
        src_amp = self.source.get_wavelet(0)

        # integrate the source wavelet, keep consistent with the non-reciprocity case
        src_amp_int = integrate.cumtrapz(
            src_amp, axis=-1, initial=0, dx=self.source.dt
        ) / (-2.0 * self.dx)

        # set up new source according to the receiver type before reciprocity
        rec_type = self.receiver.get_type(unique=True)

        if rec_type == ["vx"]:
            self.recip_type = "pr-vx"

            rec_loc = self.receiver.get_loc("vx")
            for irec in range(rec_loc.shape[0]):
                source.add_source(rec_loc[irec], src_amp_int, "vx")

        elif rec_type == ["vz"]:
            self.recip_type = "pr-vz"

            rec_loc = self.receiver.get_loc("vz")
            for irec in range(rec_loc.shape[0]):
                source.add_source(rec_loc[irec], src_amp_int, "vz")

        elif rec_type == ["vx", "vz"] or rec_type == ["vz", "vx"]:
            self.recip_type = "pr-vx-vz"

            # set up the source for vx
            rec_loc_vx = self.receiver.get_loc("vx")
            for irec in range(rec_loc_vx.shape[0]):
                source.add_source(rec_loc_vx[irec], src_amp_int, "vx")

            # set up the source for vz
            rec_loc_vz = self.receiver.get_loc("vz")
            for irec in range(rec_loc_vz.shape[0]):
                source.add_source(rec_loc_vz[irec], src_amp_int, "vz")

        elif rec_type == ["das"]:
            self.recip_type = "pr-das"

            # get all the receiver coordinates
            gauge_len = self.receiver.cable.gauge_len
            chann_num = self.receiver.cable.chann_num
            rec_loc = self.receiver.cable.get_rec_loc()[:, [0, 2]]
            tangent = self.receiver.cable.get_tangent()[:, [0, 2]]

            # loop over all the channels and add the source
            for ic in range(chann_num):
                channl_neg = 2 * ic
                channl_pos = 2 * ic + 1
                source.add_source(
                    rec_loc[channl_pos],
                    src_amp_int * tangent[channl_pos, 0] / gauge_len,
                    "vx",
                )
                source.add_source(
                    rec_loc[channl_pos],
                    src_amp_int * tangent[channl_pos, 1] / gauge_len,
                    "vz",
                )
                source.add_source(
                    rec_loc[channl_neg],
                    -src_amp_int * tangent[channl_neg, 0] / gauge_len,
                    "vx",
                )
                source.add_source(
                    rec_loc[channl_neg],
                    -src_amp_int * tangent[channl_neg, 1] / gauge_len,
                    "vz",
                )

        else:
            raise ValueError("Reciprocity Error: unsupported receiver type:", rec_type)

        # set new source and receiver for reciprocity
        self.source = source
        self.receiver = receiver

    def __set_source(self) -> None:
        """Set the source"""

        # parameters for source
        nt = self.source.nt
        dt = self.source.dt
        dx = self.dx
        src_num = self.source.num
        src_num_per_shot = 6

        # source locations and amplitudes for x and z force in velocity components
        source_fx_loc = np.zeros((src_num, src_num_per_shot, 2))
        source_fz_loc = np.zeros((src_num, src_num_per_shot, 2))
        source_fx_amp = np.zeros((src_num, src_num_per_shot, nt))
        source_fz_amp = np.zeros((src_num, src_num_per_shot, nt))

        for isrc in range(src_num):
            # duplicate the source locations for assigning dipole source
            source_fx_loc[isrc, :, :] = np.tile(
                self.source.get_loc(isrc), (src_num_per_shot, 1)
            )
            source_fz_loc[isrc, :, :] = np.tile(
                self.source.get_loc(isrc), (src_num_per_shot, 1)
            )

            # assign source amplitudes based on source type
            src_type = self.source.get_type(isrc)
            src_amp = self.source.get_wavelet(isrc)
            src_amp_int = integrate.cumtrapz(src_amp, axis=-1, initial=0, dx=dt) / (
                -2.0 * dx
            )

            if src_type == "vx":
                source_fx_amp[isrc, 0, :] = src_amp
                # source_fx_loc[isrc, 0, 1] += 0
                # source_fx_loc[isrc, 0, 0] += 0
                # source_fx_loc[isrc, 1, 1] += 0
                # source_fx_loc[isrc, 1, 0] += dx
                # source_fx_amp[isrc, 0, :] = src_amp
                # source_fx_amp[isrc, 1, :] = src_amp


            elif src_type == "vz":
                source_fz_amp[isrc, 0, :] = src_amp
                # source_fz_loc[isrc, 0, 1] -= dx
                # source_fz_loc[isrc, 0, 0] += 0
                # source_fz_loc[isrc, 1, 1] += 0
                # source_fz_loc[isrc, 1, 0] += 0
                # source_fz_amp[isrc, 0, :] = src_amp
                # source_fz_amp[isrc, 1, :] = src_amp

            elif src_type == "pr":
                # dipole source
                source_fx_loc[isrc, 0, 0] += 0
                source_fx_loc[isrc, 0, 1] += 0
                
                source_fx_loc[isrc, 1, 0] += dx
                source_fx_loc[isrc, 1, 1] += 0
                
                source_fz_loc[isrc, 0, 0] += 0
                source_fz_loc[isrc, 0, 1] -= dx
                source_fz_loc[isrc, 1, 0] += 0
                source_fz_loc[isrc, 1, 1] += 0

                source_fx_amp[isrc, 0, :] = -1.0 * src_amp_int
                source_fx_amp[isrc, 1, :] =        src_amp_int
                source_fz_amp[isrc, 0, :] = -1.0 * src_amp_int
                source_fz_amp[isrc, 1, :] =        src_amp_int

            elif src_type == "mt":
                # Moment tensor source
                sm = self.source.get_moment_tensor(isrc)
                sm = sm / (self.dx * self.dx * self.dx)

                source_fx_loc[isrc, 0, 0] += 0
                source_fx_loc[isrc, 0, 1] += 0
                source_fx_loc[isrc, 1, 0] += dx
                source_fx_loc[isrc, 1, 1] += 0
                source_fx_loc[isrc, 2, 0] += 0
                source_fx_loc[isrc, 2, 1] += dx
                source_fx_loc[isrc, 3, 0] += dx
                source_fx_loc[isrc, 3, 1] += dx
                source_fx_loc[isrc, 4, 0] += 0
                source_fx_loc[isrc, 4, 1] -= dx
                source_fx_loc[isrc, 5, 0] += dx
                source_fx_loc[isrc, 5, 1] -= dx

                source_fz_loc[isrc, 0, 0] += dx
                source_fz_loc[isrc, 0, 1] += 0
                source_fz_loc[isrc, 1, 0] += dx
                source_fz_loc[isrc, 1, 1] -= dx
                source_fz_loc[isrc, 2, 0] -= dx
                source_fz_loc[isrc, 2, 1] += 0
                source_fz_loc[isrc, 3, 0] -= dx
                source_fz_loc[isrc, 3, 1] -= dx
                source_fz_loc[isrc, 4, 0] += 0
                source_fz_loc[isrc, 4, 1] -= dx
                source_fz_loc[isrc, 5, 0] += 0
                source_fz_loc[isrc, 5, 1] += 0

                # add Mxx
                source_fx_amp[isrc, 0, :] = -sm[0, 0] * src_amp
                source_fx_amp[isrc, 1, :] =  sm[0, 0] * src_amp
                # TODO: check the index below   
                # add Mxz
                source_fx_amp[isrc, 2, :] =  sm[0, 1] * src_amp / 4.0
                source_fx_amp[isrc, 3, :] =  sm[0, 1] * src_amp / 4.0
                source_fx_amp[isrc, 4, :] = -sm[0, 1] * src_amp / 4.0
                source_fx_amp[isrc, 5, :] = -sm[0, 1] * src_amp / 4.0
                # add Mxz
                source_fz_amp[isrc, 0, :] =  sm[2, 1] * src_amp / 4.0
                source_fz_amp[isrc, 1, :] =  sm[2, 1] * src_amp / 4.0
                source_fz_amp[isrc, 2, :] = -sm[2, 1] * src_amp / 4.0
                source_fz_amp[isrc, 3, :] = -sm[2, 1] * src_amp / 4.0
                # add Mzz
                source_fz_amp[isrc, 4, :] = -sm[2, 2] * src_amp
                source_fz_amp[isrc, 5, :] =  sm[2, 2] * src_amp

            else:
                raise RuntimeError("Unknown source type: %s" % src_type)

        # set as attributes
        if self.simultaneous:
            self.source_fx_loc = source_fx_loc.reshape(
                (1, src_num_per_shot * src_num, 2)
            )
            self.source_fz_loc = source_fz_loc.reshape(
                (1, src_num_per_shot * src_num, 2)
            )
            self.source_fx_amp = source_fx_amp.reshape(
                (1, src_num_per_shot * src_num, nt)
            )
            self.source_fz_amp = source_fz_amp.reshape(
                (1, src_num_per_shot * src_num, nt)
            )

        elif self.reciprocity and self.recip_type == "pr-das":
            # there are 4 sources for each shot in the reciprocal acquisition of DAS
            self.source_fx_loc = source_fx_loc.reshape((-1, src_num_per_shot * 4, 2))
            self.source_fz_loc = source_fz_loc.reshape((-1, src_num_per_shot * 4, 2))
            self.source_fx_amp = source_fx_amp.reshape((-1, src_num_per_shot * 4, nt))
            self.source_fz_amp = source_fz_amp.reshape((-1, src_num_per_shot * 4, nt))

        else:
            self.source_fx_loc = source_fx_loc
            self.source_fz_loc = source_fz_loc
            self.source_fx_amp = source_fx_amp
            self.source_fz_amp = source_fz_amp

    def __set_receiver(self) -> None:
        """Set the receiver"""
        # assign receiver location if exists
        receiver_pr_loc = self.receiver.get_loc("pr")
        receiver_vx_loc = self.receiver.get_loc("vx")
        receiver_vz_loc = self.receiver.get_loc("vz")
        receiver_das_loc = self.receiver.get_loc("das")

        # count the number of receivers per shot
        n_receivers_pr_per_shot = receiver_pr_loc.shape[0]
        n_receivers_vx_per_shot = receiver_vx_loc.shape[0]
        n_receivers_vz_per_shot = receiver_vz_loc.shape[0]
        n_receivers_das_per_shot = receiver_das_loc.shape[0]

        # concatenate the DAS receivers to the velocity receivers
        # require both vx and vz receivers for DAS
        receiver_vx_loc = np.concatenate((receiver_vx_loc, receiver_das_loc), axis=0)
        receiver_vz_loc = np.concatenate((receiver_vz_loc, receiver_das_loc), axis=0)

        # expand to all sources, assuming the same receivers for all shots
        if self.simultaneous:
            src_num = 1

        elif self.reciprocity and self.recip_type == "pr-das":
            # there are 4 sources for each shot in the reciprocal acquisition of DAS
            src_num = self.source.num // 4

        else:
            src_num = self.source.num

        self.n_receivers_pr_per_shot = n_receivers_pr_per_shot
        self.n_receivers_vx_per_shot = n_receivers_vx_per_shot
        self.n_receivers_vz_per_shot = n_receivers_vz_per_shot
        self.n_receivers_das_per_shot = n_receivers_das_per_shot
        self.receiver_pr_loc = np.tile(receiver_pr_loc, (src_num, 1, 1))
        self.receiver_vx_loc = np.tile(receiver_vx_loc, (src_num, 1, 1))
        self.receiver_vz_loc = np.tile(receiver_vz_loc, (src_num, 1, 1))


    def record_data(self, rec_pr: torch.Tensor, 
                    rec_vz: torch.Tensor, 
                    rec_vx: torch.Tensor) -> dict:
        """Record data for different components

        Parameters
        ----------
        rec_pr : torch.Tensor
            Pressure data (nshots, nrec_pr, nt)
        rec_vz : torch.Tensor
            Vz data (nshots, nrec_vz, nt)
        rec_vx : torch.Tensor
            Vx data (nshots, nrec_vx, nt)
        Returns
        -------
        dict
            Data dictionary for recording data for different components
        """

        # data dictionary for recording data
        data = {}


        # for the non-reciprocity case
        if not self.reciprocity:
            # record pressure data, if any
            if self.n_receivers_pr_per_shot > 0:
                data["pr"] = -1.0 * rec_pr

            # record vx data, if any
            if self.n_receivers_vx_per_shot > 0:
                data["vx"] = rec_vx[:, : self.n_receivers_vx_per_shot, :]

            # record vz data, if any
            if self.n_receivers_vz_per_shot > 0:
                data["vz"] = rec_vz[:, : self.n_receivers_vz_per_shot, :]

            # record das data, if any
            if self.n_receivers_das_per_shot:
                das_vx = rec_vx[:, -self.n_receivers_das_per_shot :, :]
                das_vz = rec_vz[:, -self.n_receivers_das_per_shot :, :]
                data["das"] = self.cable_operator.forward(das_vx, das_vz)

        # for the reciprocity case
        else:
            if self.recip_type == "pr-vx":
                indices1 = np.arange(0, self.n_receivers_vx_per_shot, 2)
                indices2 = np.arange(1, self.n_receivers_vx_per_shot, 2)
                data["vx"] = (
                    rec_vx[:, indices2, :]
                    - rec_vx[:, indices1, :]
                    + rec_vz[:, indices2, :]
                    - rec_vz[:, indices1, :]
                )

            elif self.recip_type == "pr-vz":
                indices1 = np.arange(0, self.n_receivers_vz_per_shot, 2)
                indices2 = np.arange(1, self.n_receivers_vz_per_shot, 2)
                data["vz"] = (
                    rec_vx[:, indices2, :]
                    - rec_vx[:, indices1, :]
                    + rec_vz[:, indices2, :]
                    - rec_vz[:, indices1, :]
                )

            elif self.recip_type == "pr-vx-vz":
                nshots_vx = self.source.get_loc(type="vx").shape[0]
                indices1 = np.arange(0, self.n_receivers_vx_per_shot, 2)
                indices2 = np.arange(1, self.n_receivers_vx_per_shot, 2)
                data["vx"] = (
                    rec_vx[:nshots_vx, indices2, :]
                    - rec_vx[:nshots_vx, indices1, :]
                    + rec_vz[:nshots_vx, indices2, :]
                    - rec_vz[:nshots_vx, indices1, :]
                )
                data["vz"] = (
                    rec_vx[nshots_vx:, indices2, :]
                    - rec_vx[nshots_vx:, indices1, :]
                    + rec_vz[nshots_vx:, indices2, :]
                    - rec_vz[nshots_vx:, indices1, :]
                )

            elif self.recip_type == "pr-das":
                indices1 = np.arange(0, self.n_receivers_vx_per_shot, 2)
                indices2 = np.arange(1, self.n_receivers_vx_per_shot, 2)
                data["das"] = (
                    rec_vx[:, indices2, :]
                    - rec_vx[:, indices1, :]
                    + rec_vz[:, indices2, :]
                    - rec_vz[:, indices1, :]
                )

            else:
                raise ValueError(
                    "Reciprocity Error: unsupported receiver types:", self.reciq_type
                )

        return data


    def plot(self, figsize=(8, 8)) -> None:
        """Plot the survey"""
        src_type = ["Src-Pr", "Src-Vx", "Src-Vz", "Src-MT"]
        src_pr = self.source.get_loc(type="pr")
        src_vx = self.source.get_loc(type="vx")
        src_vz = self.source.get_loc(type="vz")
        src_mt = self.source.get_loc(type="moment_tensor")

        rec_type = ["Rec-Pr", "Rec-Vx", "Rec-Vz", "Rec-DAS"]
        rec_pr = self.receiver.get_loc(type="pr")
        rec_vx = self.receiver.get_loc(type="vx")
        rec_vz = self.receiver.get_loc(type="vz")
        rec_das = self.receiver.get_loc(type="das")

        # fontsize = 14
        # plt.rcParams.update(
        #     {
        #         "axes.labelsize": fontsize,
        #         "xtick.labelsize": fontsize,
        #         "ytick.labelsize": fontsize,
        #         "legend.fontsize": fontsize,
        #         "figure.titlesize": fontsize,
        #     }
        # )

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        # plot the receiver locations
        for type, loc in zip(rec_type, [rec_pr, rec_vx, rec_vz, rec_das]):
            if loc.shape[0] > 0:
                ax.scatter(loc[:, 0], loc[:, 1], marker="^", s=20, label=type)

        # plot the source locations
        for type, loc in zip(src_type, [src_pr, src_vx, src_vz, src_mt]):
            if loc.shape[0] > 0:
                ax.scatter(loc[:, 0], loc[:, 1], marker="*", s=30, label=type)

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Seismic Survey (2D)")
        ax.invert_yaxis()
        ax.grid(linewidth=1.0, color="gray", alpha=0.3)
        ax.legend(
            loc="lower left", bbox_to_anchor=(1.05, 0.0), ncol=1, borderaxespad=0.0
        )
        plt.show()
