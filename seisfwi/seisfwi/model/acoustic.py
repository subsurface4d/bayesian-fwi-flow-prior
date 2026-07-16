import logging
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import seisfwi.defaults as defaults
from seisfwi.model.base import AbstractModel
from seisfwi.utils.operators import MaskOperator, SmoothOperator, ConstrainOperator

logger = logging.getLogger(__name__)


class AcousticModel(AbstractModel):
    """
    Acoustic model with P-wave velocity parameterization.

    Parameters
    ----------
    ox, oz : float
        Model origin in x and z directions.
    dx, dz : float
        Grid spacing in x and z directions.
    nx, nz : int
        Number of grid points.
    vp : np.ndarray or torch.Tensor
        P-wave velocity model (nz, nx).
    vp_bound : tuple, optional
        (lower, upper) bounds for vp.
    vp_grad : bool, optional
        Flag for computing vp gradient.
    free_surface : bool, optional
        Free surface flag.
    nabc : int, optional
        Absorbing boundary cells.
    mask_grad : np.ndarray or torch.Tensor, optional
        Mask for gradient muting.
    mask_water : np.ndarray or torch.Tensor, optional
        Water mask (not used yet).
    smooth_size : List[int], optional
        Smoothing kernel size [kx, kz].
    """

    def __init__(
        self,
        ox: float,
        oz: float,
        dx: float,
        dz: float,
        nx: int,
        nz: int,
        vp: np.ndarray,
        vp_bound: Optional[Tuple[float, float]] = None,
        vp_grad: Optional[bool] = False,
        free_surface: Optional[bool] = False,
        nabc: Optional[int] = 20,
        mask_grad: Optional[np.ndarray] = None,
        mask_water: Optional[np.ndarray] = None,
        smooth_size: Optional[List[int]] = None,
    ) -> None:

        super().__init__(ox, oz, dx, dz, nx, nz, free_surface, nabc)

        self.pars = ["vp"]

        # Main parameter
        self.vp = torch.as_tensor(vp, dtype=defaults.dtype, device=defaults.device)
        self.vp.requires_grad_(vp_grad)

        # Optional masks, fallback to ones
        self.mask_grad = self._to_tensor(mask_grad)
        self.mask_water = self._to_tensor(mask_water, fill_value=0.0, dtype=torch.bool)

        # Bounds and gradient
        self.lower_bound["vp"] = vp_bound[0] if vp_bound else self.vp.min().item()
        self.upper_bound["vp"] = vp_bound[1] if vp_bound else self.vp.max().item()
        self.requires_grad["vp"] = vp_grad
        self.smooth_size = smooth_size

        self.check_dims()
        self.check_bounds()

        # Operators for constraints, smoothing, masking
        self.constrain_op = ConstrainOperator
        self.smooth_op = (
            SmoothOperator(smooth_size[0], smooth_size[1])
            if smooth_size else None
        )
        self.mask_op = MaskOperator(self.mask_grad)

        # Log model info
        self._logging_info()
        
        
    def _logging_info(self) -> str:
        """Log model information for debugging and tracking. """
        
        logger.debug(
            f"AcousticModel: {self.nz} x {self.nx} grid points, "
            f"dx={self.dx}, dz={self.dz}, ox={self.ox}, oz={self.oz}"
        )
        logger.debug(f"Model parameters: {self.pars}")
        logger.debug(f"Model vp shape: {self.vp.shape}, dtype: {self.vp.dtype}")
        logger.debug(f"Model vp requires_grad: {self.vp.requires_grad}")
        logger.debug(
            f"vp bounds: [{self.lower_bound['vp']}, {self.upper_bound['vp']}]"
        )
        logger.debug(
            f"Using smoothing kernel size: {self.smooth_size if self.smooth_size else 'None'}"
        )
    
    def get_ndim(self) -> int:
        return self.ndim
    
    
    def forward(self) -> torch.Tensor:
        """
        Forward pass: constrain, mask, smooth.

        Note: mask must be applied before smoothing so in backward pass, the 
        gradient is first smoothed and then masked.

        Returns
        -------
        torch.Tensor
        """
        # Constrain vp to bounds
        vp = self.constrain_op(
            self.vp, self.lower_bound["vp"], self.upper_bound["vp"]
        )
        
        # Apply mask if specified
        vp = self.mask_op(vp)

        # Apply smoothing if specified
        if self.smooth_op is not None:
            vp = self.smooth_op(vp)

        return vp


    def set_model_vector(self, m_vec: torch.Tensor) -> None:
        """
        Reset vp from flattened tensor.

        Parameters
        ----------
        m_vec : torch.Tensor
            Flattened model vector.
        """
        if not isinstance(m_vec, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(m_vec)}")

        ndim = self.ndim
        if m_vec.numel() != ndim:
            raise ValueError(f"Expected vector size {ndim}, got {m_vec.numel()}" )

        if m_vec.dtype != defaults.dtype or m_vec.device != defaults.device:
            m_vec = m_vec.to(dtype=defaults.dtype, device=defaults.device)

        # print range
        # print(m_vec.min().item(), m_vec.max().item())
        
        self.vp = m_vec.view(self.nz, self.nx)
        self.vp.grad = None

        # Ensure vp requires grad if set
        # self.vp.retain_grad()

    def get_model_vector(self):
        """
        Return flattened vp vector.

        Returns
        -------
        torch.Tensor
        """
        return self.vp.detach().flatten()
    
    
    def get_gradient_vector(self) -> torch.Tensor:
        """
        Return flattened vp gradient.

        Returns
        -------
        torch.Tensor
        """
        if self.vp.grad is None:
            logging.warning("vp gradient is None, returning zero vector.")
            return torch.zeros_like(self.vp).reshape(-1)
        return self.vp.grad.detach()[:, None]
    

class AcousticModelVAE(AbstractModel):
    """
    Acoustic model with P-wave velocity parameterization using Variational Autoencoder (VAE).

    Parameters
    ----------
    ox, oz : float
        Model origin in x and z directions.
    dx, dz : float
        Grid spacing in x and z directions.
    nx, nz : int
        Number of grid points.
    pca : sklearn.decomposition.PCA
        PCA model fitted to the training data.
    vp : np.ndarray
        The mean P-wave velocity model (nz, nx).
    vp_bound : tuple, optional
        (lower, upper) bounds for vp.
    vp_grad : bool, optional
        Flag for computing vp gradient.
    free_surface : bool, optional
        Free surface flag.
    nabc : int, optional
        Absorbing boundary cells.
    mask_grad : np.ndarray or torch.Tensor, optional
        Mask for gradient muting.
    mask_water : np.ndarray or torch.Tensor, optional
        Water mask (not used yet).
    smooth_size : List[int], optional
        Smoothing kernel size [kx, kz].
    """

    def __init__(
        self,
        ox: float,
        oz: float,
        dx: float,
        dz: float,
        nx: int,
        nz: int,
        VAE: torch.nn.Module,
        rock_physics_params: dict,
        vp: np.ndarray,
        latent_grad: Optional[bool] = False,
        latent_bound: Optional[Tuple[float, float]] = None,
        free_surface: Optional[bool] = False,
        nabc: Optional[int] = 20,
        mask_grad: Optional[np.ndarray] = None,
        mask_water: Optional[np.ndarray] = None,
        smooth_size: Optional[List[int]] = None,
    ) -> None:

        super().__init__(ox, oz, dx, dz, nx, nz, free_surface, nabc)

        self.pars = ["latent_z"]

        # Main parameter
        self.vp = torch.as_tensor(vp, dtype=defaults.dtype, device=defaults.device)

        # Optional masks, fallback to ones
        self.mask_grad = self._to_tensor(mask_grad)
        self.mask_water = self._to_tensor(mask_water, fill_value=0.0, dtype=torch.bool)

        # Bounds and gradient flags
        self.lower_bound["latent_z"] = latent_bound[0] if latent_bound else -10
        self.upper_bound["latent_z"] = latent_bound[1] if latent_bound else  10
        self.requires_grad["latent_z"] = False
        self.vp_grad = False
        self.smooth_size = smooth_size

        # VAE parameters
        self._set_VAE(VAE, latent_grad)

        # Rock Physics model
        self._set_RockPhysics(rock_physics_params)

        self.check_dims()
        self.check_bounds()

        # Operators for constraints, smoothing, masking
        self.constrain_op = ConstrainOperator
        self.smooth_op = (
            SmoothOperator(smooth_size[0], smooth_size[1])
            if smooth_size else None
        )
        self.mask_op = MaskOperator(self.mask_grad)

        # Log model info
        self._logging_info()
        
        
    def _logging_info(self) -> str:
        """Log model information for debugging and tracking. """
        
        logger.debug(
            f"AcousticModel: {self.nz} x {self.nx} grid points, "
            f"dx={self.dx}, dz={self.dz}, ox={self.ox}, oz={self.oz}"
        )
        logger.debug(f"Model parameters: {self.pars}")
        logger.debug(f"Model VAE shape: {self.VAE.latent_dim}")
        logger.debug(f"Model VAE requires_grad: {self.latent_z.requires_grad}")
        logger.debug(
            f"VAE Latent bounds: [{self.lower_bound['latent_z']}, {self.upper_bound['latent_z']}]"
        )
        logger.debug(
            f"Using smoothing kernel size: {self.smooth_size if self.smooth_size else 'None'}"
        )

    def _set_VAE(self, VAE, latent_grad) -> None:
        """ Set PCA parameters for the model.
        """        
        # eps = 1e-12
        if not isinstance(VAE, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(VAE)}")
        
        self.VAE = VAE
        ndim = self.VAE.latent_dim
        self.latent_z = torch.zeros(ndim, device=defaults.device, dtype=defaults.dtype)
        self.latent_z.requires_grad_(latent_grad)


    def _set_RockPhysics(self, rock_physics_params: dict) -> None:
        """ Set rock physics modeling: from saturation to P-wave velocity.
        """
        from seisfwi.model import rockphysics
        
        
        vp_res = rock_physics_params["vp_res"]
        vs_res = rock_physics_params["vs_res"]
        rho_res = rock_physics_params["rho_res"]
        K_dry = rock_physics_params["K_dry"]
        G_dry = rock_physics_params["G_dry"]
        K_min = rock_physics_params["K_min"]
        Rho_min = rock_physics_params["Rho_min"]
        K_brine = rock_physics_params["K_brine"]
        Rho_brine = rock_physics_params["Rho_brine"]
        K_co2 = rock_physics_params["K_co2"]
        Rho_co2 = rock_physics_params["Rho_co2"]
        phi = rock_physics_params["phi"]
        brie_component = rock_physics_params["brie_component"]

        # Create rock physics model
        RockPhyOp = rockphysics.RockPhysicsGassmann(vp_res, vs_res, rho_res,
                                                    K_dry, G_dry, 
                                                    K_min, Rho_min, 
                                                    K_brine, Rho_brine, 
                                                    K_co2, Rho_co2, 
                                                    phi, brie_component)
        
        # set the rock physics operator
        self.RockPhyOp = RockPhyOp


    @property
    def ndim(self) -> int:
        """Return total number of model elements."""
        return self.VAE.latent_dim
    
    def get_ndim(self) -> int:
        return self.ndim

    def generate(self, latent_z=None) -> torch.Tensor:
        """Generate P-wave velocity model from the VAE latent space."""
        
        # 1. Get latent vector
        latent_z = self.latent_z if latent_z is None else latent_z

        # 2. Decode to saturation and interpolate
        satura_generated = self.VAE.decode(latent_z)  # (1, 1, h, w)
        satura_generated = F.interpolate(satura_generated, size=(15, 179), 
                                         mode='bilinear', align_corners=False)

        # 3. Rock physics conversion: from saturation to vp change
        vp_change, _, _ = self.RockPhyOp(satura_generated.squeeze(0).squeeze(0))

        # 4. Pad to full model size at reservoir zone
        pad_top  = 260
        pad_left = 110
        pad_bottom = self.nz - pad_top - vp_change.shape[0]
        pad_right = self.nx - pad_left - vp_change.shape[1]
        assert pad_bottom >= 0 and pad_right >= 0, "Padded region exceeds model size."
        vp_generated = F.pad(vp_change, (pad_left, pad_right, pad_top, pad_bottom))

        # 5. Return the generated vp change for waveform modeling
        return vp_generated

    
    def forward(self) -> torch.Tensor:
        """
        Forward pass: constrain, mask, smooth.

        Note: mask must be applied before smoothing so in backward pass, the 
        gradient is first smoothed and then masked.

        Returns
        -------
        torch.Tensor
        """
        
        # VAE representation of P-wave velocity 
        vp_generated = self.generate()
        
        # fill the correct reseirvoir region
        vp = self.vp + vp_generated

        # Constrain vp to bounds
        # vp = self.constrain_op(vp, self.lower_bound["vp"], self.upper_bound["vp"])
        
        # Apply mask if specified
        vp = self.mask_op(vp)

        # Apply smoothing if specified
        if self.smooth_op is not None:
            vp = self.smooth_op(vp)

        return vp


    def get_model(self, par: str) -> np.ndarray:
        """Return the model as numpy array

        Parameters
        ----------
        par: str
            Model parameter name

        Returns
        -------
        model: np.ndarray
            Model array with shape (nz, nx)
        """

        model = getattr(self, par).clone()
        
        if par == "vp":
            # PCA representation of P-wave velocity 
            vp_generated = self.generate()

            model = self.vp + vp_generated
  
        elif par == "latent_z":
            # Return latent vector directly
            model = self.latent_z
  
        return model.clone().cpu().detach().numpy()



    def set_model_vector(self, m_vec: torch.Tensor) -> None:
        """
        Reset vp from flattened tensor.

        Parameters
        ----------
        m_vec : torch.Tensor
            Flattened model vector.
        """
        if not isinstance(m_vec, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(m_vec)}")

        if m_vec.numel() != self.ndim:
            raise ValueError(f"Expected vector size {self.ndim}, got {m_vec.numel()}" )

        if m_vec.dtype != defaults.dtype or m_vec.device != defaults.device:
            m_vec = m_vec.to(dtype=defaults.dtype, device=defaults.device)

        self.latent_z = m_vec
        self.latent_z.grad = None
                
        # # Ensure vp requires grad if set
        # self.vp.retain_grad()

    def get_model_vector(self):
        """
        Return flattened vp vector.

        Returns
        -------
        torch.Tensor
        """
        return self.latent_z.detach().flatten()


    def convert_latent_to_model(self, latent_z: np.ndarray, batch_size=1000) -> np.ndarray:
        """Convert latent vector(s) to the model parameters.

        Parameters
        ----------
        latent_z : np.ndarray
            Latent vector(s), shape (n_latent,) or (n_samples, n_latent).
        batch_size : int, optional
            Batch size for processing multiple latent samples, by default 500.

        Returns
        -------
        np.ndarray
            P-wave velocity model(s), shape (nz, nx) or (n_samples, nz, nx).
        """
        
        # Convert to Tensor
        latent_z = torch.as_tensor(latent_z, dtype=defaults.dtype, device=defaults.device)

        # Single latent vector
        if latent_z.ndim == 1:
            model = self.vp + self.generate(latent_z)  # (model_dim,)
            return model.reshape(self.nz, self.nx)

        # Multiple latent vectors
        else:
            from tqdm import tqdm
            n_samples = latent_z.shape[0]
            model = np.zeros((n_samples, self.nz * self.nx))

            for i in tqdm(range(n_samples)):
                model[i] = (self.vp + self.generate(latent_z[i])).detach().cpu().numpy().flatten()  # (model_dim,)
            
            return model.reshape(-1, self.nz, self.nx)





class AcousticModelSaturation(AbstractModel):
    """
    Acoustic model with P-wave velocity parameterization using Variational Autoencoder (VAE).

    Parameters
    ----------
    ox, oz : float
        Model origin in x and z directions.
    dx, dz : float
        Grid spacing in x and z directions.
    nx, nz : int
        Number of grid points.
    vp : np.ndarray
        The mean P-wave velocity model (nz, nx).
    vp_bound : tuple, optional
        (lower, upper) bounds for vp.
    vp_grad : bool, optional
        Flag for computing vp gradient.
    free_surface : bool, optional
        Free surface flag.
    nabc : int, optional
        Absorbing boundary cells.
    mask_grad : np.ndarray or torch.Tensor, optional
        Mask for gradient muting.
    mask_water : np.ndarray or torch.Tensor, optional
        Water mask (not used yet).
    smooth_size : List[int], optional
        Smoothing kernel size [kx, kz].
    """

    def __init__(
        self,
        ox: float,
        oz: float,
        dx: float,
        dz: float,
        nx: int,
        nz: int,
        rock_physics_params: dict,
        vp: np.ndarray,
        sat: np.ndarray,
        sat_grad: Optional[bool] = False,
        sat_bound: Optional[Tuple[float, float]] = None,
        free_surface: Optional[bool] = False,
        nabc: Optional[int] = 20,
        mask_grad: Optional[np.ndarray] = None,
        mask_water: Optional[np.ndarray] = None,
        smooth_size: Optional[List[int]] = None,
    ) -> None:

        super().__init__(ox, oz, dx, dz, nx, nz, free_surface, nabc)

        self.pars = ["sat"]

        self.nx_res = sat.shape[1]
        self.nz_res = sat.shape[0]

        # Main parameter
        self.vp = torch.as_tensor(vp, dtype=defaults.dtype, device=defaults.device)
        self.sat = torch.as_tensor(sat, dtype=defaults.dtype, device=defaults.device)

        # Optional masks, fallback to ones
        self.mask_grad = self._to_tensor(mask_grad)
        self.mask_water = self._to_tensor(mask_water, fill_value=0.0, dtype=torch.bool)

        # Bounds and gradient flags
        self.lower_bound["sat"] = sat_bound[0] if sat_bound else 0
        self.upper_bound["sat"] = sat_bound[1] if sat_bound else 1
        self.requires_grad["sat"] = sat_grad
        self.vp_grad = False
        self.smooth_size = smooth_size

        # Rock Physics model
        self._set_RockPhysics(rock_physics_params)

        self.check_dims()
        self.check_bounds()

        # Operators for constraints, smoothing, masking
        self.constrain_op = ConstrainOperator
        self.smooth_op = (
            SmoothOperator(smooth_size[0], smooth_size[1])
            if smooth_size else None
        )
        self.mask_op = MaskOperator(self.mask_grad)


        
    def _set_RockPhysics(self, rock_physics_params: dict) -> None:
        """ Set rock physics modeling: from saturation to P-wave velocity.
        """
        from seisfwi.model import rockphysics
        
        vp_res = rock_physics_params["vp_res"]
        vs_res = rock_physics_params["vs_res"]
        rho_res = rock_physics_params["rho_res"]
        K_dry = rock_physics_params["K_dry"]
        G_dry = rock_physics_params["G_dry"]
        K_min = rock_physics_params["K_min"]
        Rho_min = rock_physics_params["Rho_min"]
        K_brine = rock_physics_params["K_brine"]
        Rho_brine = rock_physics_params["Rho_brine"]
        K_co2 = rock_physics_params["K_co2"]
        Rho_co2 = rock_physics_params["Rho_co2"]
        phi = rock_physics_params["phi"]
        brie_component = rock_physics_params["brie_component"]

        # Create rock physics model
        RockPhyOp = rockphysics.RockPhysicsGassmann(vp_res, vs_res, rho_res,
                                                    K_dry, G_dry, 
                                                    K_min, Rho_min, 
                                                    K_brine, Rho_brine, 
                                                    K_co2, Rho_co2, 
                                                    phi, brie_component)
        
        # set the rock physics operator
        self.RockPhyOp = RockPhyOp


    @property
    def ndim(self) -> int:
        """Return total number of model elements."""
        return self.nx_res * self.nz_res
    
    def get_ndim(self) -> int:
        return self.ndim

    def rockphysics(self, sat) -> torch.Tensor:
        """Generate P-wave velocity model from the VAE latent space."""
        
        # 3. Rock physics conversion: from saturation to vp change
        vp_change, _, _ = self.RockPhyOp(sat)

        # 4. Pad to full model size at reservoir zone
        pad_top  = 260
        pad_left = 110
        pad_bottom = self.nz - pad_top - vp_change.shape[0]
        pad_right = self.nx - pad_left - vp_change.shape[1]
        assert pad_bottom >= 0 and pad_right >= 0, "Padded region exceeds model size."
        vp_change = F.pad(vp_change, (pad_left, pad_right, pad_top, pad_bottom))

        # 5. Return the generated vp change for waveform modeling
        return vp_change

    
    def forward(self) -> torch.Tensor:
        """
        Forward pass: constrain, mask, smooth.

        Note: mask must be applied before smoothing so in backward pass, the 
        gradient is first smoothed and then masked.

        Returns
        -------
        torch.Tensor
        """
        
        sat = self.sat
        
        # Saturation P-wave velocity 
        vp_co2 = self.rockphysics(sat)
        
        # fill the correct reseirvoir region
        vp = self.vp + vp_co2

        # Constrain vp to bounds
        # vp = self.constrain_op(vp, self.lower_bound["vp"], self.upper_bound["vp"])
        
        # Apply mask if specified
        vp = self.mask_op(vp)

        # Apply smoothing if specified
        if self.smooth_op is not None:
            vp = self.smooth_op(vp)

        return vp


    def get_model(self, par: str) -> np.ndarray:
        """Return the model as numpy array

        Parameters
        ----------
        par: str
            Model parameter name

        Returns
        -------
        model: np.ndarray
            Model array with shape (nz, nx)
        """

        model = getattr(self, par).clone()
        
        if par == "vp":
            # PCA representation of P-wave velocity 
            vp_generated = self.generate()

            model = self.vp + vp_generated
  
        elif par == "sat":
            # Return latent vector directly
            model = self.sat
  
        return model.clone().cpu().detach().numpy()


    def set_model_vector(self, m_vec: torch.Tensor) -> None:
        """
        Reset vp from flattened tensor.

        Parameters
        ----------
        m_vec : torch.Tensor
            Flattened model vector.
        """
        if not isinstance(m_vec, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(m_vec)}")

        if m_vec.numel() != self.ndim:
            raise ValueError(f"Expected vector size {self.ndim}, got {m_vec.numel()}" )

        if m_vec.dtype != defaults.dtype or m_vec.device != defaults.device:
            m_vec = m_vec.to(dtype=defaults.dtype, device=defaults.device)

        self.sat = m_vec.view(self.nz_res, self.nx_res)
        self.sat.grad = None
                
        # # Ensure vp requires grad if set
        # self.vp.retain_grad()

    def get_model_vector(self):
        """
        Return flattened vp vector.

        Returns
        -------
        torch.Tensor
        """
        return self.sat.detach().flatten()
    