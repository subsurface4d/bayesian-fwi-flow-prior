import numpy as np
import torch
import logging
from deepwave import scalar

import seisfwi.defaults as defaults
from seisfwi.model import AcousticModel
from seisfwi.survey import Survey, SeismicData
from seisfwi.propagator.hicks import Hicks

logger = logging.getLogger(__name__)


class AcousticPropagator(torch.nn.Module):
    """
    Acoustic finite-difference propagator for 2D isotropic pressure wave equation.
    """

    def __init__(self, model: AcousticModel, survey: Survey, data_mask=None, 
                 grad_interval=1):
        super().__init__()

        if not isinstance(model, AcousticModel):
            raise TypeError("`model` must be AcousticModel.")
        if not isinstance(survey, Survey):
            raise TypeError("`survey` must be Survey.")

        # Validate survey geometry fits model domain
        model.analyze_survey(survey)

        # Initialize survey: set grid size for interpolation kernels
        dx, dz = model.grid_size
        survey.initialize(dx, dz)

        if model.get_free_surface() and survey.reciprocity:
            raise RuntimeError("Free surface and reciprocity is not supported together!")

        self.model = model
        self.survey = survey
        self.data_mask = data_mask.to(defaults.device) if data_mask is not None else None
        self.grad_interval = grad_interval

        self.ox, self.oz = model.origin
        self.dx, self.dz = model.grid_size
        self.nx, self.nz = model.shape
        self.nt = survey.receiver.nt
        self.dt = survey.receiver.dt
        self.f0 = survey.source.f0

        # PML & boundary setup
        npml = model.get_nabc()
        self.free_surface = model.get_free_surface()
        self.pml_width = (
            [0, npml, npml, npml] if self.free_surface else [npml] * 4
        )
        self.free_surfaces = (
            [True, False, False, False] if self.free_surface else [False] * 4
        )

        # Source setup
        self.source_pr_ind, self.source_pr_amp = self._setup_source()

        # Receiver setup
        self.receiver_pr_ind, self.receiver_pr_hicks = self._setup_receiver()

    def _setup_source(self):
        """Correct and prepare source locations + amplitudes."""
        loc = self.survey.source_fx_loc[:, :, [1, 0]] - np.array([self.oz, self.ox])
        pr_loc = loc[:, :1, :]
        pr_amp = self.survey.source_fx_amp[:, :1] * -1.0

        if self.survey.interpolation:
            ind = torch.tensor(pr_loc / self.dx, dtype=torch.float32, device=defaults.device)
            amp = torch.tensor(pr_amp, dtype=defaults.dtype, device=defaults.device)
            hicks = Hicks(ind, free_surfaces=self.free_surfaces, model_shape=[self.nz, self.nx])
            return hicks.get_locations(), hicks.source(amp)
        else:
            ind = torch.tensor(pr_loc / self.dx, dtype=torch.long, device=defaults.device)
            amp = torch.tensor(pr_amp, dtype=defaults.dtype, device=defaults.device)
            return ind, amp

    def _setup_receiver(self):
        """Correct and prepare receiver locations + optional Hicks."""
        loc = self.survey.receiver_pr_loc[:, :, [1, 0]] - np.array([self.oz, self.ox])

        if self.survey.interpolation:
            ind = torch.tensor(loc / self.dx, dtype=torch.float32, device=defaults.device)
            hicks = Hicks(ind, free_surfaces=self.free_surfaces, model_shape=[self.nz, self.nx])
            return hicks.get_locations(), hicks
        else:
            ind = torch.tensor(loc / self.dx, dtype=torch.long, device=defaults.device)
            return ind, None

    def forward(self, model=None, shot_indx=None):
        """
        Forward propagate wavefield and record data.

        Parameters
        ----------
        model : AcousticModel, optional
            Updated model for current shot.
        shot_indx : slice, optional
            Subset of shots.
        grad_interval : int
            Interval for gradient calculation.

        Returns
        -------
        SeismicData
            Modeled shot data.
        """
        vp = self.model() if model is None else model()

        propagator = AcousticOperator(
            vp, self.dx, self.dt, self.f0, self.pml_width, self.grad_interval
        )

        n_shots = self.survey.source.num
        shot_indx = slice(0, n_shots) if shot_indx is None else shot_indx

        if defaults.device.type == "cuda" and self.survey.gpu_num > 1:
            propagator = torch.nn.DataParallel(propagator, device_ids=list(range(self.survey.gpu_num))).to(defaults.device)
            logger.info(f"Running propagator on {self.survey.gpu_num} GPUs.")

        fake_receiver = torch.ones(
            (n_shots if not self.survey.simultaneous else 1, 1, 2),
            dtype=torch.long, device=defaults.device
        )

        rec_pr = propagator(
            self.source_pr_amp[shot_indx] if self.source_pr_amp is not None else None,
            self.source_pr_ind[shot_indx] if self.source_pr_ind is not None else None,
            self.receiver_pr_ind[shot_indx] if self.receiver_pr_ind is not None else fake_receiver[shot_indx],
            defaults.device,
        )

        if self.survey.interpolation and self.receiver_pr_hicks is not None:
            rec_pr = self.receiver_pr_hicks.receiver(rec_pr)

        data_dict = self.survey.record_data(rec_pr, None, None)

        # Apply data mask if provided
        if self.data_mask is not None:
            logger.info("Applying data mask.")
            for key in data_dict:
                data_dict[key] *= self.data_mask[shot_indx]

        # Return SeismicData object with recorded data
        data = SeismicData(self.survey)
        data.record_data(data_dict)
        return data


class AcousticOperator(torch.nn.Module):
    """
    Thin wrapper for Deepwave scalar acoustic propagation.
    """

    def __init__(self, vp, dx, dt, f0, pml_width, grad_interval=1):
        super().__init__()
        self.vp = vp
        self.dx = dx
        self.dt = dt
        self.f0 = f0
        self.pml_width = pml_width
        self.accuracy = 4
        self.grad_interval = grad_interval
        self.freq_taper_frac = 0.1
        self.time_pad_frac = 0.1

    def forward(self, source_pr_amp, source_pr_ind, receiver_pr_ind, device):
        rec_pr = scalar(
            self.vp.to(device),
            self.dx,
            self.dt,
            source_amplitudes=source_pr_amp.to(device) if source_pr_amp is not None else None,
            source_locations=source_pr_ind,
            receiver_locations=receiver_pr_ind,
            accuracy=self.accuracy,
            pml_freq=self.f0,
            pml_width=self.pml_width,
            model_gradient_sampling_interval=self.grad_interval,
            freq_taper_frac=self.freq_taper_frac,
            time_pad_frac=self.time_pad_frac,
        )[-1]
        return rec_pr
