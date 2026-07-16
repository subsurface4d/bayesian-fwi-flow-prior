from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import warnings
import logging
from abc import ABC, abstractmethod
import seisfwi.defaults as defaults

logger = logging.getLogger(__name__)


class AbstractModel(torch.nn.Module, ABC):
    """
    Abstract base class for Earth model in seisfwi.

    This defines the common interface and operations for seismic model,
    including spatial grid, parameter bounds, basic math operations, and
    utilities for saving/loading. Subclasses must implement the `forward`
    method to output physical properties for a wave equation propagator.

    Attributes
    ----------
    ox : float
        Origin in x-direction (m).
    oz : float
        Origin in z-direction (m).
    dx : float
        Grid spacing in x-direction (m).
    dz : float
        Grid spacing in z-direction (m).
    nx : int
        Number of grid points in x-direction.
    nz : int
        Number of grid points in z-direction.
    free_surface : bool
        Whether to apply free surface boundary conditions.
    nabc : int
        Number of absorbing boundary cells.
    pars : List[str]
        Model parameter names.
    """

    def __init__(
        self,
        ox: float,
        oz: float,
        dx: float,
        dz: float,
        nx: int,
        nz: int,
        free_surface: bool,
        nabc: int,
    ) -> None:
        """
        Initialize common model parameters.

        Parameters
        ----------
        ox : float
            Origin in x-direction (m).
        oz : float
            Origin in z-direction (m).
        dx : float
            Grid spacing in x-direction (m).
        dz : float
            Grid spacing in z-direction (m).
        nx : int
            Number of grid points in x-direction.
        nz : int
            Number of grid points in z-direction.
        free_surface : bool
            Free surface boundary condition flag.
        nabc : int
            Number of absorbing boundary grid.
        """
        super().__init__()

        assert dx == dz, "Grid spacing dx and dz must be the same."

        self.ox = ox
        self.oz = oz
        self.dx = dx
        self.dz = dz
        self.nx = nx
        self.nz = nz
        self.free_surface = free_surface
        self.nabc = nabc

        self.x = np.arange(nx) * dx + ox
        self.z = np.arange(nz) * dz + oz

        self.pars: List[str] = []
        self.model = {}
        self.upper_bound = {}
        self.lower_bound = {}
        self.requires_grad = {}

        self.mask_grad = None
        self.mask_water = None
        
        self.pars_check = ['vp', 'vs', 'rho', 'mask_grad', 'mask_water']
        

        # logger.info(f"Initialized AbstractModel: {self}")

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the model.

        Returns
        -------
        str
            Multi-line summary of model grid, parameters, and bounds.
        """
        info = f"Earth model with parameters {self.pars}:\n"
        for par in self.pars:
            par_min = self.get_model(par).min()
            par_max = self.get_model(par).max()
            requires_grad = self.requires_grad.get(par, False)
            lower_bound = self.lower_bound.get(par)
            upper_bound = self.upper_bound.get(par)
            info += (
                f"  Model {par:>8s}: {par_min:8.2f} - {par_max:8.2f} {defaults.units[par]:6s}, "
                f"requires_grad = {requires_grad}, "
                f"constrain bound: {lower_bound:8.2f} - {upper_bound:8.2f} {defaults.units[par]:6s}\n"
            )

        info += f"  Model orig: ox = {self.ox:6.2f}, oz = {self.oz:6.2f} m\n"
        info += f"  Model grid: dx = {self.dx:6.2f}, dz = {self.dz:6.2f} m\n"
        info += f"  Model dims: nx = {self.nx:6d}, nz = {self.nz:6d}    (Total: {self.nx * self.nz * len(self.pars)}) \n"
        info += f"  Free surface: {self.free_surface} with nabc = {self.nabc}\n"
        info += f"  Device on {defaults.device} with dtype {defaults.dtype}\n"
        return info

    def copy(self) -> "AbstractModel":
        """
        Create a deep copy of this model.

        Returns
        -------
        AbstractModel
            Cloned model instance.
        """
        return self.clone(self, grad=False)

    def __sub__(self, other: "AbstractModel") -> "AbstractModel":
        """
        Subtract another model from this one.

        Parameters
        ----------
        other : AbstractModel
            Another model to subtract.

        Returns
        -------
        AbstractModel
            Resulting model after subtraction.
        """
        self.check_same_space(self, other)
        result = self.copy()
        for par in self.pars:
            # set attribute in result
            model = self._to_tensor(self.get_model(par) - other.get_model(par))
            setattr(result, par, model)
            
        return result

    def _to_tensor(self, arr, fill_value=1.0, dtype=None):
        """Helper to ensure arrays are torch tensors with defaults."""
        
        shape = (self.nz, self.nx)
        device = defaults.device
        dtype = defaults.dtype if dtype is None else dtype

        if arr is None:
            if dtype == torch.bool:
                return torch.full(shape, bool(fill_value), dtype=dtype, device=device)
            else:
                return torch.full(shape, fill_value, dtype=dtype, device=device)

        tensor = torch.as_tensor(arr, device=device)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor

    def check_dims(self) -> None:
        """Check the provided model dimensions are legal
        """
        model_list = self.pars + ['mask_grad', 'mask_water']
    
        for par in model_list:
            logger.debug(f"Checking dimensions for parameter {par}")
            if par in self.pars_check:
                assert (
                    self.get_model(par).shape == (self.nz, self.nx)
                ), "Model dimensions must be (nz, nx)"


    @staticmethod
    def check_same_space(model_a: "AbstractModel", model_b: "AbstractModel") -> None:
        """
        Ensure that two model share the same grid and parameters.

        Parameters
        ----------
        model_a : AbstractModel
            First model.
        model_b : AbstractModel
            Second model.

        Raises
        ------
        ValueError
            If any core attributes do not match.
        """
        if not isinstance(model_b, AbstractModel):
            raise TypeError("Both model must be AbstractModel instances.")
        attrs = ['pars', 'ox', 'oz', 'dx', 'dz', 'nx', 'nz', 'free_surface', 'nabc']
        for attr in attrs:
            if getattr(model_a, attr) != getattr(model_b, attr):
                raise ValueError(f"model mismatch on {attr}.")

    def check_bounds(self) -> None:
        """
        Validate and adjust parameter bounds if needed.
        """
        eps = defaults.eps
        
        for par in self.pars:
            lb, ub = self.get_bound(par)
            model = self.get_model(par)
            try:
                non_zero_min = model[model != 0].min()
            except:
                non_zero_min = 0.0
            if lb > ub:
                raise ValueError(f"Lower bound >= upper bound for {par}")

            if lb > non_zero_min:
                warnings.warn(f"Lower bound {lb:.2f} > min {non_zero_min:.2f} for {par}. Adjusted.")
                self.lower_bound[par] = float(non_zero_min - eps)

            if ub < model.max():
                warnings.warn(f"Upper bound {ub:2f} < max {model.max():2f} for {par}. Adjusted.")
                self.upper_bound[par] = float(model.max() + eps)

    @abstractmethod
    def forward(self) -> Tensor:
        """
        Must output physical properties needed for propagation.
        """
        raise NotImplementedError


    def set_model_vector(self, model_vector: torch.Tensor) -> None:
        """
        Reset the entire model using a flat 1D PyTorch tensor.

        This method should reshape slices of the input vector to match each
        parameter's shape and update the parameter tensors in place.

        Parameters
        ----------
        model_vector : torch.Tensor
            Flattened tensor containing new model parameter values.
        """
        raise NotImplementedError("This method must be implemented in the subclass.")

    def get_model_vector(self) -> torch.Tensor:
        """
        Return all model parameters concatenated as a single 1D PyTorch tensor.

        This is useful for optimization routines that expect a flat parameter vector.

        Returns
        -------
        torch.Tensor
            Flattened model parameter vector.
        """
        raise NotImplementedError("This method must be implemented in the subclass.")

    # def get_gradient_vector(self) -> torch.Tensor:
    #     """
    #     Return the gradient of the entire model as a single 1D PyTorch tensor.

    #     This is useful for checking gradient consistency or passing gradients
    #     to custom optimizers.

    #     Returns
    #     -------
    #     torch.Tensor
    #         Flattened gradient vector.
    #     """
    #     raise NotImplementedError("This method must be implemented in the subclass.")



    # ---------------- Properties ----------------

    @property
    def origin(self) -> Tuple[float, float]:
        """Return model origin (ox, oz)."""
        return self.ox, self.oz

    @property
    def shape(self) -> Tuple[int, int]:
        """Return grid dimensions (nx, nz)."""
        return self.nx, self.nz

    @property
    def grid_size(self) -> Tuple[float, float]:
        """Return grid spacing (dx, dz)."""
        return self.dx, self.dz

    @property
    def ndim(self) -> int:
        """Return total number of model elements."""
        return len(self.pars) * self.nx * self.nz
    
    @property
    def dim_shape(self) -> Tuple[int, int, int]:
        """Return model shape as (npar, nz, nx)."""
        return (len(self.pars), self.nz, self.nx)

    def get_model(self, par: str) -> Tensor:
        """
        Get model parameter tensor.

        Parameters
        ----------
        par : str
            Parameter name.

        Returns
        -------
        Tensor
            Model parameter.
        """
        model_list = self.pars + ['mask_grad', 'mask_water']
        if par not in model_list:
            raise ValueError(f"Parameter {par} not in model.")
        
        logger.debug(f"Getting model parameter {par}")
        
        arr = getattr(self, par).detach().cpu().numpy()
        return arr

    def get_bound(self, par: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get bounds for a parameter.

        Parameters
        ----------
        par : str
            Parameter name.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            (Lower bound, upper bound)
        """
        return self.lower_bound.get(par), self.upper_bound.get(par)

    def get_bound_vector(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get bounds for all parameters as flat tensors.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (Lower bounds vector, upper bounds vector)
        """
        # Build lower and upper bounds vectors for the entire model
        vector_shape = self.ndim

        lb, ub = zip(*[
            (
                np.full(vector_shape, self.get_bound(par)[0], dtype=np.float32),
                np.full(vector_shape, self.get_bound(par)[1], dtype=np.float32)
            )
            for par in self.get_pars()
        ])

        lb = np.concatenate(lb)
        ub = np.concatenate(ub)
        
        return lb, ub


    def get_grad(self, par: str) -> np.ndarray:
        """
        Get gradient as numpy array.

        Parameters
        ----------
        par : str
            Parameter name.

        Returns
        -------
        np.ndarray
            Gradient array.
        """
        m = getattr(self, par)
        return m.grad.cpu().detach().numpy() if m.grad is not None else np.zeros(m.shape)

    def get_pars(self) -> List[str]:
        """
        Get list of model parameter names.

        Returns
        -------
        List[str]
            List of parameter names.
        """
        return self.pars.copy()

    def get_free_surface(self) -> bool:
        """
        Get free surface flag.

        Returns
        -------
        bool
            True if free surface is enabled, False otherwise.
        """
        return self.free_surface

    def get_nabc(self) -> int:
        """Return the number of absorbing boundary cells

        Returns
        -------
        nabc: int
            Number of absorbing boundary cells
        """

        return self.nabc
        
    def get_clone_data(self) -> Tuple[Tuple, dict]:
        """
        Package model state for cloning.

        Returns
        -------
        Tuple
            (args, kwargs) for constructor.
        """
        args = (self.ox, self.oz, self.dx, self.dz, self.nx, self.nz)
        kwargs = {f"{par}": self.get_model(par) for par in self.pars + ["mask_grad", "mask_water"]}
        
        for par in self.pars:
            kwargs[f"{par}_bound"] = self.get_bound(par)
            kwargs[f"{par}_grad"] = self.requires_grad.get(par, False)

        kwargs["free_surface"] = self.free_surface
        kwargs["nabc"] = self.nabc
        return args, kwargs

    def save(self, filename: str) -> None:
        """
        Save model state to file.

        Parameters
        ----------
        filename : str
            Output filename.
        """
        args, kwargs = self.get_clone_data()
        np.savez(filename, *args, **kwargs)
        logger.info(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename: str, grad: bool = False, verbose: bool = False):
        """
        Load model from file.

        Parameters
        ----------
        filename : str
            Filename to load.
        grad : bool, optional
            Whether to enable gradients. Default is False.
        verbose : bool, optional
            Log details if True.

        Returns
        -------
        AbstractModel
            Loaded model instance.
        """
        data = np.load(filename, allow_pickle=True)
        num_file = 6  # adjust if needed
        args = tuple(data[f"arr_{i}"].item() for i in range(num_file))
        kwargs = {k: data[k] for k in data.files if not k.startswith("arr_")}
  
        for k in kwargs:
            if k.endswith("_grad") and k != "mask_grad":
                kwargs[k] = grad
        
        # convert bounds into a list
        for k in kwargs:
            if k.endswith("_bound"):
                kwargs[k] = list(kwargs[k]) if isinstance(kwargs[k], np.ndarray) else kwargs[k]
        
        for k in ["free_surface", "nabc"]:
            kwargs[k] = kwargs[k].item()
        
        model = cls(*args, **kwargs)
        if verbose:
            logger.info(f"Model loaded from {filename}")
        return model

    @classmethod
    def clone(cls, instance: "AbstractModel", grad: bool = False):
        """
        Clone a model instance.

        Parameters
        ----------
        instance : AbstractModel
            Instance to clone.
        grad : bool, optional
            Whether gradients should be enabled in clone.

        Returns
        -------
        AbstractModel
            Cloned model.
        """
        if not isinstance(instance, AbstractModel):
            raise TypeError("Must clone an AbstractModel instance.")
        args, kwargs = instance.get_clone_data()
        for par in instance.pars:
            kwargs[f"{par}_grad"] = grad
        return cls(*args, **kwargs)
    

    Survey = None  # Placeholder type annotation; define your Survey class elsewhere.

    def plot(
        self,
        survey: Optional[Survey] = None,
        pars: Optional[List[str]] = None,
        grad: Optional[bool] = False,
        orientation: Optional[str] = "vertical",
        sym_clip: Optional[bool] = False,
        grid_on: Optional[bool] = False,
        cmap_range: Optional[dict] = None,
        pclip: Optional[float] = 99.99,
        save_path: Optional[str] = None,
        add_label: Optional[bool] = False,
        xlim: Optional[List[float]] = None,
        zlim: Optional[List[float]] = None,
        axhline: Optional[float] = None,
        figsize=(10, 6),
        dpi = 150,
        **kwargs
    ) -> None:
        """
        Plot the model parameters and optional gradients.

        Parameters
        ----------
        survey : Survey, optional
            Survey object for overlaying source/receiver locations.
        pars : List[str], optional
            List of model parameters to plot. Defaults to all.
        grad : bool, optional
            Plot the gradient instead of the model. Default is False.
        orientation : str, optional
            'vertical' or 'horizontal' arrangement. Default is 'horizontal'.
        sym_clip : bool, optional
            Clip color limits symmetrically. Default is False.
        grid_on : bool, optional
            Show grid lines. Default is False.
        cmap_range : dict, optional
            Manually specify colormap ranges per parameter.
        clip : float, optional
            Clipping percentile for color scale. Default is 99.99.
        save_path : str, optional
            File path to save the figure.
        add_label : bool, optional
            Add subplot labels a), b), c).
        xlim : List[float], optional
            X-axis limits.
        zlim : List[float], optional
            Z-axis limits.
        axhline : float, optional
            Depth to draw a horizontal line.
        kwargs : dict
            Passed to matplotlib.pyplot.imshow.
        """
 

        pars = self.get_pars() if pars is None else pars
        if not isinstance(pars, list):
            pars = [pars]

        cmap = kwargs.get("cmap", "jet")
        fontsize = kwargs.get("fontsize", 14)
        aspect = kwargs.get("aspect", "auto")
        labels = ['a)', 'b)', 'c)']
        extent = [self.x[0], self.x[-1], self.z[-1], self.z[0]]

        fig = plt.figure(figsize=figsize, dpi=dpi)

        for i, par in enumerate(pars):
            if orientation == "vertical":
                ax = fig.add_subplot(len(pars), 1, i + 1)
            elif orientation == "horizontal":
                ax = fig.add_subplot(1, len(pars), i + 1)
            else:
                raise ValueError(f"Orientation '{orientation}' not supported.")

            if not grad:
                data = self.get_model(par)
                if sym_clip:
                    vmax = np.percentile(np.abs(data), pclip)
                    vmin = -vmax
                else:
                    if cmap_range is not None:
                        vmin, vmax = cmap_range[par]
                    else:
                        vmin, vmax = data.min(), data.max()
            else:
                data = self.get_grad(par)
                vmax = np.percentile(np.abs(data), pclip)
                vmin = -vmax

            ax.imshow(data, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
            title = f"{'GRAD-' if grad else ''}{par.upper()}"
            ax.set_title(title, fontsize=fontsize)

            if axhline is not None:
                ax.axhline(axhline, color='k', linestyle='--', linewidth=1.0)
            if xlim is not None:
                ax.set_xlim(xlim)
            if zlim is not None:
                ax.set_ylim(zlim[1], zlim[0])

            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Depth (m)")
            ax.set_aspect(aspect)
            if grid_on:
                ax.grid(True)
            if add_label:
                ax.text(-0.17, 1.1, labels[i], transform=ax.transAxes, fontsize=fontsize, va='top', fontweight='bold')

            # pad = 0.025 if orientation == "vertical" else 0.08
            cbar = fig.colorbar(ax.images[0], ax=ax, orientation=orientation, shrink=0.9)
            cbar.ax.set_ylabel(f'{defaults.units[par]}', rotation=0, fontsize=fontsize, labelpad=40)
            cbar.ax.yaxis.set_label_position("right")

            if survey is not None:
                src_type = ['Src-Pr', 'Src-Vx', 'Src-Vz', 'Src-MT']
                rec_type = ['Rec-Pr', 'Rec-Vx', 'Rec-Vz', 'Rec-DAS']

                for type, loc in zip(rec_type, [
                    survey.receiver.get_loc(type='pr'),
                    survey.receiver.get_loc(type='vx'),
                    survey.receiver.get_loc(type='vz'),
                    survey.receiver.get_loc(type='das')]):
                    if loc is not None and loc.shape[0] > 0:
                        if type == 'Rec-DAS':
                            ax.plot(loc[:, 0], loc[:, 1], 'k-', label=type, linewidth=2)
                        else:
                            ax.scatter(loc[:, 0], loc[:, 1], marker='^', s=20, c='k', label=type)

                for type, loc in zip(src_type, [
                    survey.source.get_loc(type='pr'),
                    survey.source.get_loc(type='vx'),
                    survey.source.get_loc(type='vz'),
                    survey.source.get_loc(type='mt')]):
                    if loc is not None and loc.shape[0] > 0:
                        ax.scatter(loc[:, 0], loc[:, 1], marker='*', s=30, c='r', label=type)

                ax.legend(loc='upper right', fontsize=fontsize-4, framealpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_well_log(
        self,
        x: Optional[float] = None,
        save_path: Optional[str] = None,
        add_label: Optional[bool] = False,
        figsize=(12, 8),
        dpi = 150,
        **kwargs
    ) -> None:
        """
        Plot a well log at a specified x-coordinate.

        Parameters
        ----------
        x : float, optional
            X-coordinate for the vertical profile.
        save_path : str, optional
            Path to save the figure.
        add_label : bool, optional
            Add subplot labels.
        kwargs : dict
            Passed to matplotlib.pyplot.plot.
        """

        labels = ['a)', 'b)', 'c)']
        parms = {
            "color": kwargs.get("color", "k"),
            "linewidth": kwargs.get("linewidth", 2),
            "linestyle": kwargs.get("linestyle", "-"),
        }
        fontsize = kwargs.get("fontsize", 14)

        if x is None:
            x = self.ox + self.dx * self.nx / 2

        idx = int((x - self.ox) / self.dx)
        fig = plt.figure(figsize=figsize, dpi=dpi)

        for i, par in enumerate(self.pars):
            m = self.get_model(par)
            ax = fig.add_subplot(1, 3, i + 1)
            ax.plot(m[:, idx], self.z, **parms)
            ax.set_xlabel(f"{par.upper()} (${defaults.units[par]}$)")
            ax.set_ylabel("Depth (m)")
            ax.invert_yaxis()
            ax.set_title(f'At x = {x:.2f} m')
            if add_label:
                ax.text(-0.17, 1.1, labels[i], transform=ax.transAxes,
                        fontsize=fontsize, va='top', fontweight='bold')
            ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

            ax.set_ylim(self.z[-1], self.z[0])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_mask(
        self,
        save_path: Optional[str] = None,
        fontsize: Optional[int] = 14,
        cmap: Optional[str] = "binary",
        aspect: Optional[str] = "equal",
        orientation: str = "vertical",
        figsize: Optional[Tuple[int, int]] = (8, 4),
        dpi=150,
    ) -> None:
        """
        Plot the mask applied to gradients.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.
        fontsize : int, optional
            Font size.
        cmap : str, optional
            Colormap.
        aspect : str, optional
            Aspect ratio for imshow.
        orientation : str, optional
            Orientation for colorbar.
        figsize : Tuple[int, int], optional
            Figure size.
        """

        parms = {
            "cmap": cmap,
            "aspect": aspect,
            "extent": [self.x[0], self.x[-1], self.z[-1], self.z[0]],
        }
        mask_grad = self.mask_grad.detach().cpu().numpy() 
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mask_grad, **parms)
        ax.set_title('Mask Applied to Gradient', fontsize=fontsize)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")

        cbar = fig.colorbar(ax.images[0], ax=ax, orientation=orientation)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_survey(self, survey: Survey) -> None:
        """
        Check that the survey's sources and receivers are within the model range.

        Parameters
        ----------
        survey : Survey
            Survey object.

        Raises
        ------
        RuntimeError
            If any locations are out of bounds.
        """
        src_types = set(survey.source.get_type())
        for t in src_types:
            loc = survey.source.get_loc(type=t)
            if (loc[:, 0].min() <= self.x.min() or loc[:, 0].max() >= self.x.max()
                    or loc[:, 1].min() <= self.z.min() or loc[:, 1].max() >= self.z.max()):
                raise RuntimeError('Survey Error: source location out of model range')

        rec_types = set(survey.receiver.get_type())
        for t in rec_types:
            loc = survey.receiver.get_loc(type=t)
            if (loc[:, 0].min() <= self.x.min() or loc[:, 0].max() >= self.x.max()
                    or loc[:, 1].min() <= self.z.min() or loc[:, 1].max() >= self.z.max()):
                raise RuntimeError(f'Survey Error: {t} receiver location out of model range')

        logger.info("Survey analysis completed: all sources and receivers within model range.")





# import ipywidgets as _widgets
# from IPython.display import display as _display

# def _widget_data(self) -> dict:
#     """Return the widget data

#     Returns
#     -------
#     widget_data: dict
#         Dictionary of the widget data
#     """

#     widget_data = {}

#     widget_data["Model"] = {
#         "Origin": (self.ox, self.oz),
#         "Shape": (self.nz, self.nx),
#         "Grid": (self.dx, self.dz),
#         "Free surface": self.free_surface,
#         "nabc": self.nabc,
#     }

#     for par in self.pars:
#         widget_data[par] = {
#             "shape": self.model[par].shape,
#             "min": self.lower_bound[par],
#             "max": self.upper_bound[par],
#             "grad": self.grad[par],
#             "unit": defaults.units[par],
#         }

#     return widget_data


# def __repr_html__(self):

#     # Start building the HTML string
#     default_layout = _widgets.Layout(padding="20px")

#     widget_data = self._widget_data()

#     # Helper function to make a Tab from a Python Dictionary
#     def dictionary_to_widget(dictionary):
#         left_column = _widgets.VBox(
#             [_widgets.Label(str(key)) for key in dictionary.keys()],
#             layout=default_layout,
#         )
#         right_column = _widgets.VBox(
#             [_widgets.Label(str(key)) for key in dictionary.values()],
#             layout=default_layout,
#         )
#         return _widgets.HBox([left_column, right_column], layout=default_layout)
    
#     # Create tab object
#     tab = _widgets.Tab()

#     # Populate children with sampling data
#     tab.children = [
#         dictionary_to_widget(panel_data) for panel_data in widget_data.values()
#     ]
#     panel_headings = [key for key in widget_data.keys()]
#     for i in range(len(tab.children)):
#         tab.set_title(i, panel_headings[i])

#     # Return results, or print and return nothing.

#     _display(tab)

#     return ""


# def print_results(self) -> None:
#     """Print Jupyter widget from `_repr_html_()` to stdout."""
#     print(self.__repr_html__())
