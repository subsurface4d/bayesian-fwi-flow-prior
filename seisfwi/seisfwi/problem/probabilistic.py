import logging
import os
import json
from typing import Dict, Optional, Union

import torch
import numpy as np
import arviz as az
from pathlib import Path
from tqdm import tqdm

import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.infer.autoguide.initialization import init_to_value
from pyro.infer import RandomWalkKernel, HMC, NUTS, MCMC

from seisfwi.model import AcousticModel
from seisfwi.propagator import AcousticPropagator
from seisfwi.survey import SeismicData
import seisfwi.defaults as defaults

logger = logging.getLogger(__name__)


class ProbabilisticFWI(dist.TorchDistribution):
    """
    Full-Waveform Inversion (FWI) likelihood implemented as a Pyro distribution.
    """
    
    # required for Pyro
    arg_constraints = {}
    support = constraints.real_vector

    def __init__(
        self,
        propagator: AcousticPropagator,
        model: AcousticModel,
        loss_fn: torch.nn.Module,
        dobs: SeismicData,
        weight: Optional[Dict[str, float]] = None,
        prior: Optional[dist.Distribution] = None,
        temp: float = 1.0,
        grad_interval: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        propagator : AcousticPropagator | ElasticPropagator
            The forward propagator.
        model : AcousticModel | ElasticModel
            The Earth model.
        loss_fn : torch.nn.Module
            Misfit loss function.
        dobs : SeismicData
            Observed seismic data.
        weight : dict, optional
            Data component weights.
        prior : pyro.distributions.Distribution, optional
            Prior distribution for model parameters.
        temp : float, optional
            Temperature for simulated annealing.
        grad_interval : int, optional
            Gradient computation interval.
        Note
        ----
        Key methods:
        - sample: sample the model parameters
        - log_prob: compute the log probability of the model parameters
        """
                 
        self.propagator = propagator
        self.model = model
        self.loss_fn = loss_fn
        self.dobs = dobs
        self.weight = weight or {comp: 1.0 for comp in self.dobs.data.keys()}
        self.prior = prior
        self.temp = temp
        self.grad_interval = grad_interval
        self.sigma2 = None # variance of the noise, initialized to 1, to be estimated later
 
        self.cached_log_prob = []
        self.cached_prior_min = []
        self.cached_prior_max = []
        self.count = 0

        self._validate_inputs()

        # pyro's TorchDistribution requires an event_shape
        ndim = self.model.get_ndim()
        super().__init__(event_shape=torch.Size([ndim,]))


    def _validate_inputs(self) -> None:
        """Internal validation of inputs and weights.
        """
        
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("`model` must be a torch.nn.Module.")
        if not isinstance(self.propagator, torch.nn.Module):
            raise TypeError("`propagator` must be a torch.nn.Module.")
        if not isinstance(self.loss_fn, torch.nn.Module):
            raise TypeError("`loss_fn` must be a torch.nn.Module.")
        if not isinstance(self.dobs, SeismicData):
            raise TypeError("`dobs` must be a SeismicData instance.")
        if not isinstance(self.weight, dict):
            raise TypeError("`weight` must be a dict.")
        if self.prior is not None and not isinstance(self.prior, dist.Distribution):
            raise TypeError("`prior` must be a pyro.distributions.Distribution.")

        valid_components = {"vx", "vz", "pr", "das"}
        for comp, w in self.weight.items():
            if comp not in valid_components:
                raise ValueError(f"Invalid component `{comp}`. Valid: {valid_components}")
            if comp not in self.dobs.rec_type:
                raise ValueError(f"Component `{comp}` not present in observed data.")
            if w < 0:
                raise ValueError(f"Weight for `{comp}` must be non-negative.")

        nt = self.propagator.survey.source.nt
        if nt % self.grad_interval != 0:
            raise ValueError(f"`grad_interval` ({self.grad_interval}) must divide `nt` ({nt}).")


    def sample(self, sample_shape=torch.Size([])) -> torch.Tensor:
        """Sample model parameters from the prior if provided, else from the model.
        """
        
        if self.prior:
            return self.prior.sample(sample_shape)
        return self.model.get_model_vector()


    def compute_data_misfit(self, m: torch.Tensor) -> torch.Tensor:
        """Compute the misfit between synthetic and observed data.
        """
        
        # model synthetic data with new model parameters
        self.model.set_model_vector(m)
        dsyn = self.propagator(self.model)

        loss_data = torch.tensor(0.0, dtype=defaults.dtype, device=defaults.device)
        for comp, w in self.weight.items():
            loss_comp = self.loss_fn(dsyn.data[comp], self.dobs.data[comp])
            loss_data += w * loss_comp
            logger.debug(f"Misfit for `{comp}`: weight={w}, loss={loss_comp.item():.4f}")

        return loss_data
        
        
    def log_prob(self, m: torch.Tensor) -> torch.Tensor:
        """Compute the negative log probability of the model parameters.
        """
        # if no sigma2 is set, estimate it from the initial model
        if self.sigma2 is None:
            self.estimate_sigma()
        
        # data loss
        loss_data = self.compute_data_misfit(m)
    
        # model loss (if any)
        # loss_model = self.model.get_model_loss()
        loss_model = 0.0
        
        loss = loss_data + loss_model

        # probability is p(x) = exp(-loss/temp), so log[p(x)] = -loss/temp
        return -loss / self.sigma2 / self.temp
    
    
    def estimate_sigma(self) -> torch.Tensor:
        """Estimate sigma^2 from residuals of the initial model.
        """

        logger.info("Estimating sigma^2 from initial model...")
        
        # compute the residuals using the initial model
        m = self.model.get_model_vector()
        loss_data = self.compute_data_misfit(m)

        # set sigma2 to the variance of the residuals
        self.sigma2 = loss_data
        
        logger.info(f"Estimated sigma^2: {self.sigma2:.4f}")
        


    def run(
        self,
        pars: Dict[str, float],
        method: str = "hmc",
        compute_log_prob: bool = False,
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Run the FWI inference workflow.

        Parameters
        ----------
        pars : Dict
            The parameters for the inference, including:
            rng_seed : int, optional.
            num_warmup : int, optional.
            num_samples : int, optional.
            step_size : float, optional.
            num_steps : int, optional.
            adapt_step_size : bool, optional.
            adapt_mass_matrix : bool, optional.
            target_accept_prob : float, optional.
        method : str, optional
            The inference method, by default "hmc". Options are "mcmc", "hmc", "nuts"
        compute_log_prob : bool, optional
            Whether to compute the log probability of the model parameters, by default False
        save_path : str, optional
            The path to save the inference results, by default None [not saving]
        """
        
        self.pars = pars
        self.method = method.lower()
        self.save_path = Path(save_path) if save_path else None

        self._check_pars()

        print('********************************************************')
        print(f'               {method.upper()} INFERENCE ALGORITHM        ')
        print('********************************************************\n')

        # # define the pyro model
        # def pyro_model(likelihood: 'ProbabilisticFWI', prior: Optional[dist.Distribution] = None):
            
        #     # Sample model in an unconstrained space
        #     if prior is None:
        #         pyro.sample("m", likelihood)
                
        #     # Sample model from the prior distribution
        #     else:
        #         m = pyro.sample("m", prior)
        #         print("Piror distribution:", m.shape, m.min(), m.max())
        #         pyro.sample("obs", likelihood, obs=m)

        # # Use a partial function to pass additional arguments to pyro_model
        # conditioned_model = lambda: pyro_model(self, self.prior)

        # define the pyro model
        def pyro_model(likelihood: 'ProbabilisticFWI'):
            
            # Sample model in an unconstrained space
            if self.prior is None:
                pyro.sample("m", likelihood)
                
            # Sample model from the prior distribution
            else:
                m = pyro.sample("m", self.prior)
                pyro.sample("obs", likelihood, obs=m)

        # Use a partial function to pass additional arguments to pyro_model
        conditioned_model = lambda: pyro_model(self)

        # Initialize the model with the initial model vector
        init_strategy=init_to_value(values={"m": self.model.get_model_vector()})

        # MCMC: Markov Chain Monte Carlo with Random Walk
        if self.method == "mcmc":
            # for mcmc, use default values
            kernel = RandomWalkKernel(conditioned_model, 
                                    init_step_size=0.5, 
                                    target_accept_prob=0.234,
                                    )
            
        # HMC: Hamiltonian Monte Carlo
        elif self.method == "hmc":
            kernel = HMC(
                conditioned_model,
                step_size=self.pars.get("step_size", 1.0),
                num_steps=self.pars.get("num_steps", 10),
                adapt_step_size=self.pars.get("adapt_step_size", True),
                adapt_mass_matrix=self.pars.get("adapt_mass_matrix", False),
                target_accept_prob=self.pars.get("target_accept_prob", 0.65),
                init_strategy=init_strategy,
            )

        # NUTS: No-U-Turn Sampler
        elif self.method == "nuts":
            kernel = NUTS(
                conditioned_model, 
                step_size=self.pars.get("step_size", 1.0), 
                adapt_step_size=self.pars.get("adapt_step_size", True),
                adapt_mass_matrix=self.pars.get("adapt_mass_matrix", False),
                target_accept_prob=self.pars.get("target_accept_prob", 0.65),
                init_strategy=init_strategy,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Initialize the MCMC kernel
        mcmc = MCMC(kernel, 
                    warmup_steps=self.pars['num_warmup'], 
                    num_samples=self.pars['num_samples'],)
        
        # Run the inference
        mcmc.run()

        # compute the log probability if required. Pyro does not store this value.
        log_prob = None
        if compute_log_prob:
            log_prob = self.compute_log_prob(mcmc)

        # get prior
        if self.prior is not None:
            prior = self.prior.sample((self.pars['num_samples'],)).cpu().numpy()
            
        # save the inference results if required
        if save_path:
            self.save(mcmc, log_prob, prior)

        torch.cuda.empty_cache()

        return mcmc, log_prob


    def _check_pars(self) -> None:
        """Validate inference parameters.
        """
        
        required = ["num_warmup", "num_samples"]
        for key in required:
            if key not in self.pars:
                raise ValueError(f"Missing required parameter: {key}")

        if self.method not in ["mcmc", "hmc", "nuts"]:
            raise ValueError(f"Unknown method: {self.method}")

        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        logger.info(f"Running inference with method: {self.method}")
        logger.info(f"Prior: {self.prior}")
        logger.info(f"Save Path: {self.save_path}")
        for k, v in self.pars.items():
            logger.info(f"{k:>15}: {v}")
        

    def save(self, mcmc: MCMC, log_prob: Optional[np.ndarray] = None, 
             prior: Optional[np.ndarray] = None) -> None:
        """
        Save MCMC inference results to disk in ArviZ NetCDF format, along with optional log-probability
        and prior arrays, and parameter settings in JSON format.
        """
        if self.method == "mcmc":
            logger.warning("MCMC method does not support saving ArviZ output.")
            return

        self.save_path.mkdir(parents=True, exist_ok=True)

        filename = self.save_path / f"Posterior-{self.method}-num-{self.pars['num_samples']}-temp-{self.temp}.nc"
        
        # Save MCMC results to ArviZ NetCDF
        mcmc_az = az.from_pyro(mcmc, log_likelihood=False)
        az.to_netcdf(mcmc_az, filename)

        # Save log probability array if provided
        if log_prob is not None:
            los_prob_filename = filename.with_name(filename.name.replace("Posterior", "LogProb").replace(".nc", ".npy"))    
            np.save(los_prob_filename, log_prob)

        # Save prior array if provided
        if prior is not None:
            prior_filename = filename.with_name(filename.name.replace("Posterior", "Prior").replace(".nc", ".npy"))
            np.save(prior_filename, prior)

        # Save parameters as JSON
        json_file = filename.with_suffix(".json")
        with open(json_file, "w") as f:
            json.dump(self.pars, f, indent=2)

        logger.info(f"Inference results saved to {filename}")
        print(f"Inference results saved to {filename}\n")
        

    def compute_log_prob(self, mcmc: MCMC) -> np.ndarray:
        """Compute the log probability of all MCMC samples.
        """
        
        logger.info("Computing log probability for samples...")
        
        samples = mcmc.get_samples()["m"]
        log_probs = []
        for s in tqdm(samples, desc="Computing log probabilities"):
            with torch.no_grad():
                log_probs.append(self.log_prob(s).cpu().numpy())
        return np.array(log_probs)

