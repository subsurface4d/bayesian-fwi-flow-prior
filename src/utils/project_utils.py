import numpy as np
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
import torch


def build_pca_prior_model(sat, n_components=None):
    """
    Build PCA prior generator from an ensemble of 2D models.

    Parameters:
    - sat: ndarray of shape (num_models, nz, nx)
    - n_components: int or None, number of PCA components to keep

    Returns:
    - pca: trained PCA object
    - mean_model: 1D mean model (nz*nx,)
    - generator: function that returns a new prior sample of shape (nz, nx)
    """
    num_models, nz, nx = sat.shape
    X = sat.reshape(num_models, nz * nx)  # Flatten to (num_models, model_size)

    # Mean-center the data
    mean_model = X.mean(axis=0)

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X - mean_model)

    def generator(latent_space= None):
        """
        Generate a new model sample from the PCA prior.
        """
        # Sample PCA coefficients from standard normal
        if latent_space is None:
            latent_space = np.random.randn(pca.n_components_)
        
        coeffs = latent_space * np.sqrt(pca.explained_variance_)
        sample_flat = mean_model + np.dot(coeffs, pca.components_)
        return sample_flat.reshape(nz, nx)

    return pca, mean_model.reshape(nz, nx), generator




def structural_similarity_index(image1, image2):
    # Constants for SSIM calculation
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    
    # Mean of image1 and image2
    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    
    # Variance of image1 and image2
    var1 = np.var(image1)
    var2 = np.var(image2)
    
    # Covariance between image1 and image2
    covar = np.cov(image1, image2)
    
    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * covar + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2)
    ssim = numerator / denominator
    
    return np.mean(ssim)



# class PCA(object):
#     def __init__(self, nc=1, nr=1, l=1):
#         self.l = l
#         self.nc = nc
#         self.nr = nr
#         self.xm = np.zeros((nc, 1))
#         self.usig = np.zeros((nc, l))
#         self.data_matrix = None
#         self.sig = None
#         self.u = None

#     def input_usig(self, usig=None, u=None, sig=None):
#         if usig is not None:
#             assert usig.shape == (self.nc, self.l)
#             self.usig = usig
#         elif u is not None and sig is not None:
#             assert u.shape == (self.nc, self.l)
#             assert sig.shape[0] == self.l
#             if sig.shape == (self.l, ):
#                 sig = sig[:,None]
#             self.u = u
#             self.sig = sig
#             self.usig = np.dot(self.u, np.diag(self.sig[:,0]))

#     def input_xm(self, xm):
#         assert xm.shape[0] == self.nc
#         if xm.shape == (self.nc, ):
#             xm = xm[:, None]
#         self.xm = xm

#     def construct_pca(self, x):
#         assert x.shape == (self.nc, self.nr)
#         self.data_matrix = x
#         self.xm = np.mean(x, axis=1)[:, None]
#         y = 1./(np.sqrt(float(self.nr - 1.))) * (x - self.xm)
#         self.u, self.sig, _ = np.linalg.svd(y, full_matrices=False)
#         self.u = self.u[:, :self.l]
#         self.sig = self.sig[:self.l, None]
#         self.usig = np.dot(self.u, np.diag(self.sig[:,0]))

#     def generate_pca_realization(self, xi, dim=None):
#         if dim is None:
#             assert xi.shape[0] == self.l
#             if xi.shape == (self.l, ):
#                 xi = xi[:,None]
#             return self.usig.dot(xi) + self.xm
#         else:
#             assert xi.shape[0] == dim
#             if xi.shape == (dim, ):
#                 xi = xi[:, None]
#             return self.usig[:, :dim].dot(xi) + self.xm

#     def get_xi(self, m, dim=None):
#         assert self.u is not None, "Input or calculate U matrix to obtain reconstructed xi"
#         assert m.shape[0] == self.nc
        
#         if m.shape == (self.nc, ):
#             m = m[:,None]
#         if dim is None:
#             xi = self.u.T.dot(m - self.xm)/self.sig
#         else:
#             xi = self.u[:, :dim].T.dot(m - self.xm) / self.sig[:dim]
#         return xi



# def compute_autocorr(x, max_lag=100):
#     """
#     Compute the autocorrelation function (ACF) for a 1D array.

#     Parameters
#     ----------
#     x : np.ndarray
#         Input 1D array (e.g., MCMC samples).
#     max_lag : int
#         Maximum lag to compute autocorrelation.

#     Returns
#     -------
#     acf : np.ndarray
#         Autocorrelation values for lags 0 to max_lag.
#     """
#     x = np.asarray(x)
#     n = len(x)
#     x_mean = np.mean(x)
#     x_var = np.var(x)

#     max_lag = min(max_lag, n - 1)

#     if x_var == 0:
#         return np.zeros(max_lag + 1)

#     acf = np.empty(max_lag + 1)
#     for lag in range(max_lag + 1):
#         cov = np.sum((x[:n - lag] - x_mean) * (x[lag:] - x_mean)) / (n - lag)
#         acf[lag] = cov / x_var

#     return acf

def compute_autocorr(x, max_lag=400):
    """
    Compute autocorrelation function (ACF) for a 1D MCMC chain
    up to a specified max_lag.

    Parameters
    ----------
    x : array-like, shape (N,)
        MCMC samples for a single parameter.
    max_lag : int
        Maximum lag for ACF computation and return.

    Returns
    -------
    lags : ndarray, shape (max_lag+1,)
        Array of lag values: [0, 1, ..., max_lag]
    acf  : ndarray, shape (max_lag+1,)
        Autocorrelation values, with acf[0] = 1.
    """
    x = np.asarray(x)
    n = x.size
    x_centered = x - x.mean()

    # --- FFT-based autocorrelation ---
    fft_len = 1 << (2 * n - 1).bit_length()  # next power of 2 for efficiency
    f = np.fft.fft(x_centered, fft_len)
    acf_full = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf_full /= acf_full[0]  # normalize so acf(0) = 1

    # --- Return only up to max_lag ---
    max_lag = min(max_lag, n - 1)
    lags = np.arange(max_lag + 1)
    acf = acf_full[:max_lag + 1]

    return lags, acf


# Integrated Autocorrelation Time (IAT)
def integrated_autocorr_time(x, max_lag=1000, min_rho=0.0):
    """
    Compute integrated autocorrelation time τ for a 1D MCMC chain.
    
    Parameters
    ----------
    x : array-like
        MCMC chain samples.
    max_lag : int
        Maximum lag to include in summation.
    min_rho : float
        Stop summing once rho_k < min_rho (e.g., 0.0).
    
    Returns
    -------
    tau : float
        Integrated autocorrelation time estimate.
    """
    acf = compute_autocorr(x)
    n = x.size

    max_lag = min(max_lag, n-1)
    
    tau = 1.0  # includes lag 0

    for k in range(1, max_lag+1):
        if acf[k] < min_rho:
            break
        tau += 2.0 * acf[k]
    
    return tau


# ------------------------------------------------------------
# 3. Effective Sample Size (ESS)
# ------------------------------------------------------------
def ess_1d(x, max_lag=1000, min_rho=0.0):
    """
    Compute effective sample size for a 1D chain x.
    """
    n = len(x)
    tau = integrated_autocorr_time(x, max_lag=max_lag, min_rho=min_rho)
    ess = n / tau
    return ess, tau



def rockphysic_inversion(rockphy, vp_obs, max_iter = 150, device='cuda', verbose=False):

    vp_obs = torch.tensor(vp_obs, dtype=torch.float32, device=device)

    # default setting for inversion
    reg_lambda = 1

    # initial model
    saturation = torch.zeros_like(vp_obs, requires_grad=True, device=device)
    misfit = torch.nn.MSELoss()
    
    # Create LBFGS optimizer
    # optimizer = LBFGS([saturation])
    optimizer = torch.optim.Adam([saturation], lr=1e-2)  # You might need to adjust the learning rate
    
    # Define the closure function for LBFGS
    def closure():
        optimizer.zero_grad()
        # forward
        vp_syn, _, _ = rockphy(saturation)
    
        # data misfit
        data_loss = misfit(vp_obs, vp_syn)
    
        # regularization
        reg_term = reg_lambda * torch.sum(torch.relu(-saturation))
    
        loss = data_loss + reg_term
        loss.backward()
        
        return loss
    
    # Optimization loop
    for i in range(max_iter):
        loss = optimizer.step(closure)
    
        # Projection step to enforce SCO2_inv within [0, 1]
        with torch.no_grad():
            saturation.clamp_(min=0, max=1)
        
        if verbose:
            if i%10 == 0 or i == max_iter-1:
                print(f"Iteration {i + 1:04d}/{max_iter}: Loss = {loss.item():.2f}")

    return saturation.detach().cpu().numpy()




def structural_similarity_index(image1, image2):
    
    return ssim(image1, image2, data_range=image1.max() - image2.min())


def rmse(image1, image2):
    return np.sqrt(np.mean((image1 - image2) ** 2))

