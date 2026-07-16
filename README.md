# Bayesian Full-Waveform Monitoring of CO2 Storage

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/DOI-10.1029%2F2025JH001190-blue.svg)](https://doi.org/10.1029/2025JH001190)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Research code for **"Bayesian Full-Waveform Monitoring of CO2 Storage With Fluid-Flow Priors via Generative Modeling"** (Li, Wang, Durlofsky & Biondi, *JGR: Machine Learning and Computation*, 2026).

A Bayesian framework for time-lapse (4D) seismic monitoring of geological CO2 storage with quantified uncertainty: reservoir flow simulations generate prior CO2 saturation fields, a variational autoencoder (VAE) learns a compact latent prior, rock physics maps saturation to seismic velocity, and Hamiltonian Monte Carlo (HMC) samples the posterior — recovering the $CO_{2}$ plume *and* its uncertainty even under sparse, noisy acquisition where deterministic FWI fails. Built on the framework, which is bundled in `seisfwi/`.

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
pip install -e ./seisfwi
pip install -e .
```

The first command installs the bundled `seisfwi` framework; the second installs this project plus the scientific stack and puts `src/utils` on your path. A CUDA GPU is recommended. A `Dockerfile` is also provided (`docker build -t bayesian-fwi-flow-prior .`, then mount `model/` and `VAE/` at runtime).

## Data and pretrained VAE

The input models and the trained VAE are **not tracked in git** — download them from Google Drive:

**➡️ https://drive.google.com/drive/folders/1FsD5D3I3hqRqedKm7sasxMwF83Qhh7rd?usp=sharing**

Put the velocity/saturation arrays and the noise trace in `model/`, and the VAE checkpoint in `VAE/`. Exact filenames are listed in `model/README.md` and `VAE/README.md`.

## Usage

Run everything from the repository root — the `scripts/` runners `cd` there automatically, and each notebook's setup cell does the same and puts `src/` on the path.

The numbered files form a single pipeline: notebooks `01`–`04` prepare the models, `scripts/05_*` run the inversions, and notebooks `06`–`11` analyze the results.

| Step | Notebook / script | Purpose |
| --- | --- | --- |
| `01` | `notebooks/01-Prior-Models` | Build prior CO₂ saturation & velocity models |
| `02`–`03` | `notebooks/02,03-Prior-VAE-*` | Train and validate the VAE generative prior |
| `04` | `notebooks/04-FWI-Test` | Deterministic FWI sanity check |
| `05` | `scripts/05_run_fwi_*.sh` | Run the FWI / HMC inversions across GPUs |
| `06`–`09` | `notebooks/06-09-HMC-FWI-Results-*` | Analyze posterior results |
| `10`–`11` | `notebooks/10,11-HMC-FWI-*` | Workflow schematic & data figures |

`notebooks/revision/` holds the peer-review experiments (baseline error, sparse acquisition, survey design, FWI-VAE and regularized-FWI baselines).

**Running step `05`.** The `scripts/05_run_fwi_*.sh` runners launch the `src/` drivers across GPUs — for example:

```bash
bash scripts/05_run_fwi_survey.sh    # survey/well sweep; see also *_dense, *_test, *_sensitivity
```

> First point each script's `out_dir` — and the VAE `checkpoint_path` in `src/HMC_FWI_VAE.py` — at your own paths.

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
