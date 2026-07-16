# syntax=docker/dockerfile:1
#
# Container image for the FWI-HMC CO2-monitoring project.
# Base: official PyTorch image with CUDA (GPU-enabled torch + nvcc for compiling
# the Deepwave/seisfwi CUDA extensions). You may instead base this on the
# seisfwi image if you prefer: `FROM seisfwi/seisfwi`.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

LABEL org.opencontainers.image.title="bayesian-fwi-flow-prior" \
      org.opencontainers.image.description="Bayesian full-waveform monitoring of CO2 storage with fluid-flow priors via generative modeling" \
      org.opencontainers.image.source="https://github.com/seisfwi/bayesian-fwi-flow-prior" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# System packages: git (to install seisfwi from source) and build tools (native wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/app

# Install Python dependencies first so this layer is cached across code changes.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install -e ".[notebooks]"

# Copy the remaining project files (notebooks, scripts, docs).
COPY . .

# NOTE: large inputs (model/) and the VAE checkpoint (VAE/) are NOT included in the
# image. Download them from Google Drive (see model/README.md and VAE/README.md) and
# mount them at runtime:
#
#   docker run --gpus all -it \
#       -v "$(pwd)/model:/workspace/app/model" \
#       -v "$(pwd)/VAE:/workspace/app/VAE" \
#       bayesian-fwi-flow-prior
#
# To serve the notebooks, expose Jupyter:
#   docker run --gpus all -p 8888:8888 -it bayesian-fwi-flow-prior \
#       jupyter lab --ip 0.0.0.0 --no-browser --allow-root

CMD ["/bin/bash"]
