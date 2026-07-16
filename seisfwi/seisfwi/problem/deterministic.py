import logging
from typing import Optional, Union

import os
import torch
import numpy as np

from seisfwi.model import AcousticModel
from seisfwi.propagator import AcousticPropagator
from seisfwi.survey import SeismicData
from seisfwi.optimizer import SeiscopeOptimizer

import seisfwi.defaults as defaults

logger = logging.getLogger(__name__)


class DeterministicFWI(torch.nn.Module):
    """
    Full-Waveform Inversion (FWI) implementation.

    Attributes:
        propagator (AcousticPropagator): The forward propagator.
        model (AcousticModel): The Earth model.
        loss_fn (torch.nn.Module): Misfit loss function.
        dobs (SeismicData): Observed seismic data.
        weight (dict): Data component weights.
        regularizer (torch.nn.Module, optional): Regularization module.
        reg_loss_scale (float): Weight for the regularization term.
        grad_interval (int): Gradient computation interval.
    """

    def __init__(
        self,
        propagator: Union[AcousticPropagator],
        model: Union[AcousticModel],
        loss_fn: torch.nn.Module,
        dobs: SeismicData,
        weight: Optional[dict] = None,
        regularizer: Optional[torch.nn.Module] = None,
        reg_loss_scale: float = 0.1,
        grad_interval: int = 1,
    ) -> None:
        
        super().__init__()
        self.propagator = propagator
        self.model = model
        self.loss_fn = loss_fn
        self.dobs = dobs
        self.regularizer = regularizer
        self.reg_loss_scale = reg_loss_scale
        self.grad_interval = grad_interval
        self.data_loss_scale = 1.0

        self.weight = weight or {comp: 1.0 for comp in self.dobs.data.keys()}
        if weight is None:
            logger.info("No data weights provided. Using default weight=1.0 for all components.")

        self._validate_inputs()

    def _validate_inputs(self):
        """Internal validation of inputs and weights."""
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("`model` must be a torch.nn.Module [AcousticModel or ElasticModel].")
        if not isinstance(self.propagator, torch.nn.Module):
            raise TypeError("`propagator` must be a torch.nn.Module [AcousticPropagator or ElasticPropagator].")
        if not isinstance(self.loss_fn, torch.nn.Module):
            raise TypeError("`loss_fn` must be a torch.nn.Module.")
        if not isinstance(self.dobs, SeismicData):
            raise TypeError("`dobs` must be a SeismicData instance.")
        if not isinstance(self.weight, dict):
            raise TypeError("`weight` must be a dict.")

        valid_components = {"vx", "vz", "pr", "das"}
        for comp, w in self.weight.items():
            if comp not in valid_components:
                raise ValueError(f"Invalid component `{comp}`. Valid: {valid_components}")
            if comp not in self.dobs.rec_type:
                raise ValueError(f"Component `{comp}` not present in observed data.")
            if w < 0:
                raise ValueError(f"Weight for `{comp}` must be non-negative.")
            
        # Check if grad_interval divides nt
        nt = self.propagator.survey.source.nt
        if nt % self.grad_interval != 0:
            raise ValueError(f"`grad_interval` ({self.grad_interval}) must divide `nt` ({nt}).")


    def compute_loss_and_gradient(self,
        model_tensor,
        nshots_per_batch: int = 1,
    ) -> float:
        """
        Compute batched loss and gradient for all shots.

        Args:
            model_vector (torch.Tensor): Flattened model parameters.
            nshots_per_batch (int): Shots per GPU per batch.

        Returns:
            float: Total epoch loss.
        """
        
        self.model.set_model_vector(model_tensor)
        
        n_shots = self.propagator.survey.source.num
        n_gpus = self.propagator.survey.gpu_num
        n_batches = n_shots // (n_gpus * nshots_per_batch) + 1
        
        epoch_data_loss = 0.0        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * n_gpus * nshots_per_batch
            end_idx = min(start_idx + n_gpus * nshots_per_batch, n_shots)
            if end_idx <= start_idx:
                continue

            shot_slice = slice(start_idx, end_idx)
            dsyn = self.propagator(self.model, shot_indx=shot_slice)

            batch_loss = torch.tensor(0.0, dtype=defaults.dtype, device=defaults.device)
            for comp, w in self.weight.items():
                loss_comp = self.loss_fn(dsyn.data[comp], self.dobs.data[comp][shot_slice])
                batch_loss += w * loss_comp
                logger.debug(f"Batch `{comp}`: weight={w}, loss={loss_comp.item():.4f}")

            # Backprop for this batch
            batch_loss.backward()

            # Accumulate scalar value (detach from graph)
            epoch_data_loss += batch_loss.item()

        total_loss = epoch_data_loss
        # print(f"Data loss: {epoch_data_loss:.6f}, Reg loss: {reg_loss.item():.6f}, Total loss: {total_loss:.6f}, scale {scale}")

        return total_loss


    def run(
        self,
        optimizer: str = "torchmin",
        method: str = "l-bfgs",
        max_iter: int = 10,
        nshots_per_batch: int = 1,
        grad_scale: float = 10.0,
        conv: float = 1e-4,
        log_file: Optional[str] = None,
    ):
        """
        Run the FWI inversion workflow.

        Args:
            optimizer (str): 'torchmin' or 'seiscope'.
            method (str): Optimization method name.
            max_iter (int): Max number of iterations.
            nshots_per_batch (int): Shots per batch.
            grad_scale (float): Gradient scaling factor for seiscope.
            conv (float): Convergence criterion for seiscope.
            log_file (Optional[str]): Log file path for optimization output.
        """
        
        optimizer = optimizer.lower()
        method = method.lower()

        if optimizer not in {"torchmin", "seiscope"}:
            raise ValueError("`optimizer` must be 'torchmin' or 'seiscope'.")

        if optimizer == "torchmin" and method.lower() in {"lbfgs", "l-bfgs"}:
            method = "l-bfgs"
        elif optimizer == "seiscope" and method.lower() in {"lbfgs", "l-bfgs"}:
            method = "LBFGS"

        # mkdir for log_file if provided
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logger.info(f"Starting FWI: optimizer={optimizer}, method={method}, max_iter={max_iter}")

        if optimizer == "seiscope":
            res = {"model": [],
                   "grad": [],
                   "loss": [],}
            
            # build the lb and ub bounds if needed
            lb, ub = self.model.get_bound_vector()
                            
            # Initialize Seiscope optimizer
            optimizer = SeiscopeOptimizer(niter_max = max_iter, 
                                          conv = conv, 
                                          method = method.upper(),
                                          bound = True, 
                                          lb = lb, 
                                          ub = ub, 
                                          nls_max=6,
                                          log_file=log_file)

            # Get initial model vector as tensor
            model_tensor = self.model.get_model_vector().detach()
            model_tensor.requires_grad_(True)
            
            # Compute initial loss and gradient
            loss = self.compute_loss_and_gradient(model_tensor, nshots_per_batch)
            grad = model_tensor.grad.detach().cpu().numpy()
            
            # Scale the gradient
            self.data_loss_scale = grad_scale / np.max(np.abs(grad))
            loss *= self.data_loss_scale
            grad *= self.data_loss_scale
            grad_preco = grad.copy()
        
            # cache initial results
            res["model"].append(model_tensor.detach().cpu().numpy().copy())
            res["grad"].append(grad.copy())
            res["loss"].append(loss.item())

            while optimizer.FLAG not in {"CONV", "FAIL"}:
                # Convert tensor to numpy for seiscope
                model_numpy = model_tensor.detach().cpu().numpy()

                # Perform one optimization step
                model_numpy = optimizer.iterate(model_numpy, loss, grad, grad_preco)

                # Re-wrap as tensor for next gradient calculation
                model_tensor = torch.tensor(model_numpy, requires_grad=True, 
                                            device=defaults.device, 
                                            dtype=defaults.dtype)
                        
                if optimizer.FLAG == "GRAD":
                    loss = self.compute_loss_and_gradient(model_tensor, nshots_per_batch)
                    grad = model_tensor.grad.detach().cpu().numpy()
                    loss *= self.data_loss_scale
                    grad *= self.data_loss_scale
                    grad_preco = grad.copy()
                    
                    # cache results
                    res["model"].append(model_numpy.copy())
                    res["grad"].append(grad.copy())
                    res["loss"].append(loss.item())


            # convert final model to numpy
            res["model"] = np.array(res["model"])
            res["grad"] = np.array(res["grad"])
            res["loss"] = np.array(res["loss"])
            
            logger.info("FWI optimization completed.")
            
            return model_numpy, res

