# Notebooks

Numbered, end-to-end analysis and figure notebooks. Run them from the repository
root (e.g. `jupyter lab` launched at the root). Each notebook begins with a small
auto-added **setup cell** that changes to the repo root and puts `src/` on the
Python path, so `from utils import ...`, `model/...`, and `figures/...` resolve
regardless of where the notebook lives.

## Pipeline order

| Notebook | Purpose |
| --- | --- |
| `01-Prior-Models` | Build prior CO₂ saturation / velocity models from flow simulations |
| `02-Prior-VAE-Training` | Train the VAE generative prior |
| `03-Prior-VAE-Validation` | Validate VAE reconstruction, sampling, latent continuity |
| `04-FWI-Test` | Deterministic FWI sanity check |
| `06-HMC-FWI-Results` | Main Bayesian (HMC) monitoring results and uncertainty |
| `07-HMC-FWI-Results-Survey` | Survey-design comparison from posterior uncertainty |
| `08-HMC-FWI-Results-Waveforms` | Observed vs. predicted waveform fits |
| `09-HMC-FWI-Results-Sensitivity` | Sensitivity to prior / rock-physics assumptions |
| `10-HMC-FWI-Workflow` | Workflow schematic |
| `11-HMC-FWI-Data` | Seismic data figures |
| `00-Figure-Slides`, `11-HMC-FWI-Results-Slides` | Presentation figures |

## `revision/`

Additional experiments produced during peer review: baseline-error robustness
(`R03`), sparse acquisition and survey design (`R04`), deterministic FWI in the
VAE latent space (`R05`), and regularized deterministic FWI (`R06`), plus a VAE
retrain/validation pass (`R02`).
