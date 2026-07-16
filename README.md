# Bayesian Full-Waveform Monitoring of CO₂ Storage

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/DOI-10.1029%2F2025JH001190-blue.svg)](https://doi.org/10.1029/2025JH001190)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Research code for **"Bayesian Full-Waveform Monitoring of CO₂ Storage With Fluid-Flow Priors via Generative Modeling"** (Li, Wang, Durlofsky & Biondi, *JGR: Machine Learning and Computation*, 2026).

A Bayesian framework for time-lapse (4D) seismic monitoring of geological CO₂ storage with quantified uncertainty: reservoir flow simulations generate prior CO₂ saturation fields, a variational autoencoder (VAE) learns a compact latent prior, rock physics maps saturation to seismic velocity, and Hamiltonian Monte Carlo (HMC) samples the posterior — recovering the $CO_{2}$ plume *and* its uncertainty even under sparse, noisy acquisition where deterministic FWI fails. Built on the framework, which is bundled in `seisfwi/`.

## Repository structure

```
bayesian-fwi-flow-prior/
├── src/          FWI_VAE, FWI_REG, HMC_FWI_VAE, HMC_FWI_VAE_DD  +  utils/ (VAE, rock physics, plots)
├── seisfwi/      bundled FWI framework — install with `pip install -e ./seisfwi`
├── scripts/      bash runners (survey / well / frequency / noise / temperature sweeps)
├── notebooks/    numbered pipeline  +  revision/ (peer-review experiments)
├── model/        input data — git-ignored (download from Google Drive)
└── VAE/          pretrained VAE — git-ignored (download from Google Drive)
```

## Installation

```bash
git clone https://github.com/seisfwi/bayesian-fwi-flow-prior.git
cd bayesian-fwi-flow-prior
conda create -n bayesian-fwi python=3.10 -y && conda activate bayesian-fwi
pip install -e ./seisfwi  # bundled FWI framework
pip install -e .          # this project (add ".[notebooks]" for Jupyter)
```

The first command installs the bundled `seisfwi` framework; the second installs this project plus the scientific stack and puts `src/utils` on your path. A CUDA GPU is recommended. A `Dockerfile` is also provided (`docker build -t bayesian-fwi-flow-prior .`, then mount `model/` and `VAE/` at runtime).

## Data and pretrained VAE

The input models and the trained VAE are **not tracked in git** — download them from Google Drive:

**➡️ https://drive.google.com/drive/folders/1FsD5D3I3hqRqedKm7sasxMwF83Qhh7rd?usp=sharing**

Put the velocity/saturation arrays and the noise trace in `model/`, and the VAE checkpoint in `VAE/`. Exact filenames are listed in `model/README.md` and `VAE/README.md`.

## Usage

Run everything from the repository root — the bash runners `cd` there automatically, and each notebook has a small setup cell that does the same and puts `src/` on the path.

**Notebooks** (`notebooks/`) form the pipeline: `01` prior models → `02`/`03` VAE train/validate → `04` FWI test → `06`–`09` HMC results (main, survey, waveforms, sensitivity) → `10`/`11` workflow & data. `notebooks/revision/` holds the peer-review experiments (baseline error, sparse acquisition, survey design, FWI-VAE and regularized-FWI baselines). Launch with `jupyter lab` from the root.

**Batch runs** (`scripts/`) launch the `src/` drivers across GPUs — for example:

```bash
bash scripts/05_run_fwi_survey.sh     # HMC survey / well sweep
python src/HMC_FWI_VAE.py --output_dir results/run1 --f0 30 --src_id 2 --rec_id 1 \
    --vp_ml_file model/vp_ml_nz346_nx401_5m.npy --device cuda:0 --temp 0.025
```

> The `out_dir` in `scripts/*.sh` and `checkpoint_path` in `src/HMC_FWI_VAE.py` use absolute paths from the authors' cluster — edit them to your own (e.g. `results/`, `VAE/...pth`).

## Citation

```bibtex
@article{Li2026BayesianFWM,
  title   = {Bayesian Full-Waveform Monitoring of CO2 Storage With Fluid-Flow Priors via Generative Modeling},
  author  = {Li, Haipeng and Wang, Nanzhe and Durlofsky, Louis J. and Biondi, Biondo L.},
  journal = {Journal of Geophysical Research: Machine Learning and Computation},
  volume  = {3}, pages = {e2025JH001190}, year = {2026},
  doi     = {10.1029/2025JH001190}
}
```

## License

MIT — see [LICENSE](LICENSE). Built on [seisfwi](https://github.com/seisfwi/seisfwi), [Deepwave](https://github.com/ar4/deepwave), and [rockphypy](https://github.com/rockphysics/rockphypy).
