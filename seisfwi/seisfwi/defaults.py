"""
Default parameters and units for seisfwi.
"""

import torch

units = {
    "vp": "m/s",
    "vs": "m/s",
    "rho": "kg/m^3",
    "lam": "Pa",
    "mu": "Pa",
    "ip": "Pa·s/m",
    "is": "Pa·s/m",
    "vpvs": "",
    "ref": "kg/m^2/s",
    "rbf_theta": "m/s",
    "latent_z": 'm/s',
    "log_vp": "",
    "log_vs": "",
    "log_vpvs": "",
    "log_ip": "",
}

eps = 1e-7

# Default data type and device for tensors
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert dtype in [torch.float32, torch.float64], "dtype must be float32 or float64"
assert isinstance(device, torch.device), "device must be a torch.device"
