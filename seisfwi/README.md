<img src="./docs/logo1.png" alt="logo" style="zoom:60%;" />

[![Actions Status](https://github.com/seisfwi/seisfwi/actions/workflows/workflow.yml/badge.svg)](https://github.com/seisfwi/seisfwi/actions)
[![coverage](https://codecov.io/gh/seisfwi/seisfwi/branch/main/graph/badge.svg)](https://codecov.io/gh/seisfwi/seisfwi)
[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://seisfwi.github.io/seisfwi/)
[![supported versions](https://img.shields.io/pypi/pyversions/seisfwi.svg?label=python_versions)](https://pypi.python.org/pypi/seisfwi)
[![docs](https://badge.fury.io/py/seisfwi.svg)](https://badge.fury.io/py/seisfwi)

**SeisFWI** is a Python package that provides Full-waveform Inversion (FWI) solution for high-resolution subsurface imaging and monitoring.

It provides:

- A versatile and easy-to-use platform for researchers developing and testing new FWI algorithms.
- A efficient tool for performing computationally expensive FWI on CPUs or GPUs in parallel.
- A practical package for practitioners to perform FWI on field data with necessary pre/post processing utilities.
- A reproducible framework for users to reproduce successful synthetic and field-data FWI cases.

It solves:

$$
\begin{align}
\min_{m} \Phi(m) &= \chi(d, d^{\text{obs}}) + \beta L(m)
\\
s.t. \quad d &= \mathcal{R}\mathbf{F}(\mathbf{\theta(m)}, \mathcal{S})
\end{align}
$$

where $\Phi(m)$ is the objective function, $\chi(d, d^{\text{obs}})$ is the data misfit, $L(m)$ is the regularization term, $\mathcal{R}$ is the receiver operator, $\mathbf{F}$ is the wave equation operator, $\mathbf{\theta(m)}$ is the model parameterization operator, and $\mathcal{S}$ is the source operator. For more details, please refer to [the documentation](https://seisfwi.github.io/seisfwi/).

## Installation
```bash
# Create a new conda environment
conda create -n seisfwi python=3.9
conda activate seisfwi

# Install seisfwi
pip install -e .
```

## Citation
- Li, H., Liu, J., Mao, S., Yuan, S., Clapp, R. G., & Biondi, B. L. (2025). Fiber‐optic seismic full waveform monitoring of groundwater dynamics. Geophysical Research Letters, 52(20), e2025GL1176. https://doi.org/10.1029/2025GL117610
- Li, H., Li, J., Luo, S., Bem, T. S., Yao, H., & Huang, X. Continent‐continent Collision between the South and North China Plates Revealed by Seismic Refraction and Reflection at the Southern Segment of the Tanlu Fault Zone. Journal of Geophysical Research: Solid Earth, e2022JB025748. https://doi.org/10.1029/2022JB025748
- Li, H., Li, J., Liu, B., & Huang, X. (2021). Application of full-waveform tomography on deep seismic profiling data set for tectonic fault characterization. In First International Meeting for Applied Geoscience & Energy Expanded Abstracts (pp. 657–661). https://doi.org/10.1190/segam2021-3583190.1