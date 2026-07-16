import os
import argparse
import logging
import numpy as np
import torch
from scipy import signal
from pathlib import Path
import arviz as az

# SEISFWI imports
from seisfwi.model import AcousticModel, AcousticModelSaturation, AcousticModelVAE
from seisfwi.survey import Survey, Source, Receiver
from seisfwi.propagator import AcousticPropagator
from seisfwi.problem import DeterministicFWI, ProbabilisticFWI, L2Loss, GaussianPrior
from seisfwi.utils import wavelet, smooth2d
from seisfwi.signal import norm_traces, add_noise, bandpass_filter
import seisfwi.defaults as defaults

# VAE model
from utils.vae import VanillaVAE
from utils import RockPhysicsModel

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_survey(f0 = 60.0, src_id=1, rec_id=1, vp_ml_file='model/vp_ml_nz346_nx401_5m.npy')  -> tuple:
    """Creates the model and survey objects."""
    
    # ------------------------------------------------------------------------------
    # MODEL SETUP
    # ------------------------------------------------------------------------------
    ox, oz = 0.0, 0.0
    nx, nz = 401, 346
    dx, dz = 5.0, 5.0
    free_surface =True

    vp_bl = np.load('model/vp_bl_nz346_nx401_5m.npy')
    vp_ml = np.load(vp_ml_file)

    model_bl = AcousticModel(ox, oz, dx, dz, nx, nz, vp = vp_bl, free_surface=free_surface)
    model_ml = AcousticModel(ox, oz, dx, dz, nx, nz, vp = vp_ml, free_surface=free_surface)

    # ------------------------------------------------------------------------------
    # SURVEY SETUP
    # ------------------------------------------------------------------------------
    nt, dt, amp = 1200, 0.001, 1e7
    wvlt = -1.0 * np.diff(np.diff(wavelet(nt, dt, f0) * amp, prepend=0), prepend=0)

    # Sources
    source = Source(nt=nt, dt=dt, f0=f0)
    
    if src_id == 1:
        for isrc in range(5):
            source.add_source([400 * isrc + 200, dx], wvlt, 'pr')

    elif src_id == 2:    
        source.add_source([500, dx], wvlt, 'pr')
        
    elif src_id == 3:
        source.add_source([1000, dx], wvlt, 'pr')
    
    elif src_id == 4:    
        source.add_source([1500, dx], wvlt, 'pr')

    else:
        raise ValueError("Invalid src_id. Choose 1, 2, 3 or 4.")
        
    # Receivers
    geophone = Receiver(nt=nt, dt=dt)

    if rec_id == 1:
        for irec in range(nz-35):
            geophone.add_receiver([700, dx * (irec + 1)], 'pr')

    # Borehole receivers at two wells
    elif rec_id == 2:
        for irec in range(nz-35):
            geophone.add_receiver([700, dx * (irec + 1)], 'pr')

        for irec in range(nz-35):
            geophone.add_receiver([1300, dx * (irec + 1)], 'pr')

    else:
        raise ValueError("Invalid rec_id. Choose 1, or 2.")

    # Survey with 1 GPU
    survey = Survey(source=source, receiver=geophone, gpu_num=1)

    return model_bl, model_ml, survey



def run_workflow(args):
    
    # Argument parsing    
    output_dir = args.output_dir
    f0 = args.f0
    src_id = args.src_id
    rec_id = args.rec_id
    vp_ml_file = args.vp_ml_file
    device = args.device
    noise_snr = args.noise_snr
    temp = args.temp
        
    # Setup defaults
    path = Path(output_dir)
    os.makedirs(path, exist_ok=True)

    defaults.device = torch.device(device)
    
    # Get model, survey and data parameters
    model_bl, model_ml, survey = get_survey(f0=f0, src_id=src_id, rec_id=rec_id, vp_ml_file=vp_ml_file)
    
    ox, oz = model_bl.ox, model_bl.oz
    nx, nz = model_bl.nx, model_bl.nz
    dx, dz = model_bl.dx, model_bl.dz
    free_surface = model_bl.free_surface
    vp_bl = model_bl.get_model('vp')
    
    # ------------------------------------------------------------------------------
    # SYNTHETIC DATA
    # ------------------------------------------------------------------------------
    F = AcousticPropagator(model_bl, survey)
    dobs_bl = F(model_bl)
    dobs_ml = F(model_ml)

    # Save all data
    dobs_bl.save(path / "dobs-BL.npz")
    dobs_ml.save(path / "dobs-ML.npz")
    
    if noise_snr is not None:
        logging.info(f"Adding noise with SNR: {noise_snr}")
        # Add noise
        nt, dt = dobs_bl.nt, dobs_bl.dt
        noise = np.loadtxt('model/Noise_trace178_nt1600.dat')
        noise = norm_traces(noise)
        noise = signal.resample(noise, nt, axis=1)
        noise = bandpass_filter(noise, dt, 2, 80, order=4)

        # Make a copy to avoid modifying original data
        dobs_bl_noisy = dobs_bl.data['pr'].cpu().numpy().copy()
        dobs_ml_noisy = dobs_ml.data['pr'].cpu().numpy().copy()

        nsrc = dobs_bl_noisy.shape[0]
        nrec = dobs_bl_noisy.shape[1]
        
        for isrc in range(nsrc):
            np.random.seed(40 + isrc)
            noise_add_bl = noise[np.random.randint(0, 178, size=nrec), :]
            np.random.seed(50 + isrc)
            noise_add_ml = noise[np.random.randint(0, 178, size=nrec), :]
            # Apply noise to the selected source slice
            dobs_bl_noisy[isrc, :, :] = add_noise(dobs_bl_noisy[isrc, :, :], noise_add_bl, snr=noise_snr)
            dobs_ml_noisy[isrc, :, :] = add_noise(dobs_ml_noisy[isrc, :, :], noise_add_ml, snr=noise_snr)
            
        # put the noisy data back to dobs_bl
        dobs_bl.data['pr'] = torch.from_numpy(dobs_bl_noisy).float().cuda()
        dobs_ml.data['pr'] = torch.from_numpy(dobs_ml_noisy).float().cuda()

        dobs_bl.save(path / f"dobs-BL-Noise-SNR-{noise_snr}.npz")
        dobs_ml.save(path / f"dobs-ML-Noise-SNR-{noise_snr}.npz")


    # ------------------------------------------------------------------------------
    # DETERMINISTIC FWI SETTINGS
    # ------------------------------------------------------------------------------
    loss_fn = L2Loss
    weight = {'pr': 1.0}
    optimizer = "seiscope"
    max_iter = 100
    grad_scale = 10.0
    nshots_per_batch = 4

    # ------------------------------------------------------------------------------
    # FWI: FINITE DIFFERENCE MODEL
    # ------------------------------------------------------------------------------
    # mask_grad = np.ones_like(vp_bl)
    # mask_grad[:140,:] = 0.0
    # mask_grad = smooth2d(mask_grad, 5, 5)
    
    nz0 = 260
    nx0 = 110
    nz_res = 15
    nx_res = 179
    mask_grad = np.zeros_like(vp_bl)
    mask_grad[nz0:nz0+nz_res, nx0:nx0+nx_res] = 1.0
    
    # model_fwi = AcousticModel(
    #     ox, oz, dx, dz, nx, nz, vp_bl,
    #     vp_grad=True,
    #     vp_bound=[1000, 6000],
    #     mask_grad=mask_grad,
    #     free_surface=free_surface
    # )

    # fwi = DeterministicFWI(F, model_fwi, loss_fn, dobs_ml, weight)
    # fwi.run(
    #     optimizer=optimizer,
    #     max_iter=max_iter,
    #     grad_scale=grad_scale,
    #     nshots_per_batch=nshots_per_batch,
    #     log_file=path / "FWI.log"
    # )

    # # Save the all models
    # model_bl.save(path / "Model-BL.npz")
    # model_ml.save(path / "Model-ML.npz")
    # model_fwi.save(path / "Model-FWI.npz")

    # ------------------------------------------------------------------------------
    # ROCK PHYSICS MODEL SETUP
    # ------------------------------------------------------------------------------
    nz0, nx0 = 260, 110
    nz_res, nx_res = 15, 179
    vp_res = vp_bl[nz0:nz0+nz_res, nx0:nx0+nx_res].copy()
    rock_physics_params = RockPhysicsModel(vp_res)

    # # ------------------------------------------------------------------------------
    # # FWI WITH SATURATION PARAMETERIZATION via ROCK PHYSICS MODEL
    # # ------------------------------------------------------------------------------
    # model_fwi_sat = AcousticModelSaturation(
    #     ox, oz, dx, dz, nx, nz, 
    #     rock_physics_params,
    #     vp_bl,
    #     sat = np.zeros_like(vp_res),
    #     sat_grad=True,
    #     sat_bound=[0, 1],
    #     mask_grad=mask_grad,
    #     free_surface=free_surface)

    # fwi = DeterministicFWI(F, model_fwi_sat, loss_fn, dobs_ml, weight)
    
    # model_numpy, _ = fwi.run(
    #     optimizer=optimizer,
    #     max_iter=max_iter,
    #     grad_scale=0.1,
    #     nshots_per_batch=nshots_per_batch,
    #     log_file=path / "FWI-Sat.log"
    # )
    # np.save(path / "Model-FWI-Sat.npy", model_numpy)
    
    # ------------------------------------------------------------------------------
    # FWI WITH VAE PRIORS 
    # ------------------------------------------------------------------------------

    # Load VAE Model
    kld_weight = 0.000015
    in_channels = 1
    latent_dim = 64
    device = torch.device(device)
    checkpoint_path = f"/net/vision/scr2/haipeng/FWI-HMC/VAE/vae_latent_dim{latent_dim}_kld_weight{kld_weight}.pth"

    # Reload the model
    model_vae = VanillaVAE(in_channels=in_channels, latent_dim=latent_dim).to(defaults.device)
    model_vae.load_state_dict(torch.load(checkpoint_path, map_location=defaults.device))
    model_vae.eval()

    model_fwi_vae = AcousticModelVAE(
        ox, oz, dx, dz, nx, nz, 
        model_vae,
        rock_physics_params,
        vp_bl,
        latent_grad=True,
        latent_bound=[-10, 10],
        mask_grad=mask_grad,
        free_surface=free_surface
    )

    # set the seed for reproducibility
    torch.manual_seed(11)
    num_samples = 1
    # z_fwi = torch.randn(num_samples, model_vae.latent_dim).to(device)
    z_fwi = torch.ones(num_samples, model_vae.latent_dim).to(device)
    # z_fwi = torch.zeros(num_samples, model_vae.latent_dim).to(device)
    model_fwi_vae.set_model_vector(z_fwi)

    fwi = DeterministicFWI(F, model_fwi_vae, loss_fn, dobs_ml, weight)

    model_numpy, res = fwi.run(
        optimizer=optimizer,
        max_iter=max_iter,
        grad_scale=grad_scale,
        nshots_per_batch=nshots_per_batch,
        log_file=path / "FWI-VAE.log"
    )

    np.save(path / "Model-FWI-VAE.npy", model_numpy)
    np.save(path / "Model-FWI-VAE-Res.npy", res)




    # # ------------------------------------------------------------------------------
    # # POST-PROCESSING: STATISTICS
    # # ------------------------------------------------------------------------------
    # num_samples = hmc_params['num_samples']
    # latent_prior        = np.load(       path / f"HMC/Prior-hmc-num-{num_samples}-temp-{temp}.npy")
    # log_prob            = np.load(       path / f"HMC/LogProb-hmc-num-{num_samples}-temp-{temp}.npy")
    # latent_posterior_az = az.from_netcdf(path / f"HMC/Posterior-hmc-num-{num_samples}-temp-{temp}.nc")
    # latent_posterior    = latent_posterior_az.posterior['m'].values[0]

    # # Convert latent samples to velocity model
    # vp_posterior = model_fwi_hmc.convert_latent_to_model(latent_posterior) - vp_bl
    # vp_prior     = model_fwi_hmc.convert_latent_to_model(latent_prior) - vp_bl

    # # Prior statistics
    # vp_prior_mean = np.mean(vp_prior, axis=0)
    # vp_prior_std  = np.std(vp_prior, axis=0)
    # vp_prior_p10  = np.percentile(vp_prior, 10, axis=0)
    # vp_prior_p50  = np.percentile(vp_prior, 50, axis=0)
    # vp_prior_p90  = np.percentile(vp_prior, 90, axis=0)

    # # Posterior statistics
    # vp_posterior_mean = np.mean(vp_posterior, axis=0)
    # vp_posterior_std  = np.std(vp_posterior, axis=0)
    # vp_posterior_p10  = np.percentile(vp_posterior, 10, axis=0)
    # vp_posterior_p50  = np.percentile(vp_posterior, 50, axis=0)
    # vp_posterior_p90  = np.percentile(vp_posterior, 90, axis=0)

    # # MAP of the posterior samples
    # index  = np.argmax(log_prob)
    # vp_map = vp_posterior[index]

    # # makedir if not exist
    # (path / "Statistics").mkdir(parents=True, exist_ok=True)
    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-Prior-Mean.npy", vp_prior_mean)
    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-Prior-Std.npy", vp_prior_std)
    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-Prior-P10.npy", vp_prior_p10)
    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-Prior-P50.npy", vp_prior_p50)
    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-Prior-P90.npy", vp_prior_p90)

    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-Posterior-Mean.npy", vp_posterior_mean)
    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-Posterior-Std.npy", vp_posterior_std)
    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-Posterior-P10.npy", vp_posterior_p10)
    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-Posterior-P50.npy", vp_posterior_p50)
    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-Posterior-P90.npy", vp_posterior_p90)
    # np.save(path / f"Statistics/VP-{num_samples}-temp-{temp}-MAP.npy", vp_map)

    # np.save(path / f"Statistics/LogProb-{num_samples}-temp-{temp}.npy", log_prob)
    # logging.info(f"Statistics saved in {path / 'Statistics'}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Deterministic and Bayesian FWI with VAE priors.")

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for all files')
    parser.add_argument('--f0', type=float, default=60.0,
                        help='Central frequency of the source wavelet')
    parser.add_argument('--src_id', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Survey type')
    parser.add_argument('--rec_id', type=int, default=1, choices=[1, 2, 3, 4], help='Well configuration')
    parser.add_argument('--vp_ml_file', type=str, default='model/vp_ml_nz221_nx_401_5m_1.npy',
                        help='File path for the ML velocity model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the computations on (e.g., cuda:0, cpu)')
    parser.add_argument('--noise_snr', type=float, default=None,
                        help='Noise level for the synthetic data')
    parser.add_argument('--temp', type=float, default=0.025, help='Temperature for HMC sampling')
    args = parser.parse_args()

    print("--------------------------------------------------------------")
    print("🚀 Running SEISFWI Workflow")
    print("--------------------------------------------------------------")
    print(f" Output Directory: {args.output_dir}")
    print(f"Central Frequency: {args.f0} Hz")
    print(f"        Survey ID: {args.src_id}")
    print(f"          Well ID: {args.rec_id}")
    print(f"   Velocity Model: {args.vp_ml_file}")
    print(f"           Device: {args.device}")
    print(f"        Noise SNR: {args.noise_snr}")
    print(f"      Temperature: {args.temp}")
    print("--------------------------------------------------------------\n")

    # Run the workflow
    run_workflow(args)

    print("\n✅ Workflow completed successfully.")
