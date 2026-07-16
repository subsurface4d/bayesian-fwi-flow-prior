from rockphypy import BW
import numpy as np


def RockPhysicsModel(vp_res):
    
    # Vs, and rho from empirical relations
    vs_res  = vp_res / np.sqrt(3.0)
    rho_res = vp_res ** 0.25 * 310.0

    # modulus saturated by brine in GPa
    G_sat = rho_res * vs_res**2 / 1e9
    K_sat = rho_res * vp_res**2 / 1e9 - 4.0 / 3.0 * G_sat
    Rho_sat = rho_res.copy()
        
    minerals = {
        "quartz":   {"K":36.6, "G":45.0, "rho":2650.0},   # GPa, GPa, kg/m3
        "kfs":      {"K":57.0, "G":28.0, "rho":2620.0},   # GPa, GPa, kg/m3
        "clay":     {"K":20.9, "G": 6.9, "rho":2580.0},   # GPa, GPa, kg/m3
        "calcite":  {"K":76.8, "G":32.0, "rho":2710.0},   # GPa, GPa, kg/m3
        "dolomite": {"K":94.9, "G":45.0, "rho":2870.0},   # GPa, GPa, kg/m3
    }

    # Clean fluvial–deltaic sandstone (good φ, high k)
    mineral_frac_clean_sand = {"quartz": 0.70, "kfs": 0.10, "clay": 0.15, "calcite": 0.015, "dolomite": 0.035}

    volumes, K, G, rho = [], [], [], []
    for m, f in mineral_frac_clean_sand.items():
        volumes.append(f)
        K.append(minerals[m]["K"])
        G.append(minerals[m]["G"])
        rho.append(minerals[m]["rho"])
        
    _, _, _, _, K_min, G_min, Rho_min = vrh(volumes, K, G, rho)


    # Fluid properties
    temp = 60       # reservoir temperature, deg C
    Pp   = 14       # reservoir pressure, MPa
    salinity = 35000/1000000 # 0.011

    # Brine properties
    Rho_brine, K_brine = BW.rho_K_brine(temp, Pp, salinity)

    # CO2 properties
    G = 1.5349                          # gas gravity of CO2
    Rho_co2, K_co2 = BW.rho_K_co2(Pp, temp, G)
    
    # G_brine = 0
    # G_co2 = 0

    # g/cm3 to kg/m3
    Rho_co2   *= 1000
    Rho_brine *= 1000

    # Estimate effective porosity and dry frame moduli from baseline
    phi   = estimate_porosity_from_density(Rho_sat, Rho_min, Rho_brine)
    K_dry = K_min * (K_brine*K_min + K_brine*K_sat*phi - K_brine*K_sat - K_min*K_sat*phi) / (K_brine*K_min*phi + K_brine*K_min - K_brine*K_sat - K_min**2*phi)
    G_dry = G_sat

    print("    Brine properties: K = %.2f GPa, rho = %.4f kg/m3" % (K_brine, Rho_brine))
    print("      CO2 properties: K = %.2f GPa, rho = %.4f kg/m3" % (K_co2, Rho_co2))
    print("  Mineral properties: K = %.2f GPa, rho = %.4f kg/m3" % (K_min, Rho_min))
    print("  Estimated porosity: phi = %.4f" % (np.mean(phi)))    
    
    brie_component=2


    rock_physics_params = {
        'vp_res': vp_res,
        'vs_res': vs_res,
        'rho_res': rho_res,
        'K_dry': K_dry,
        'G_dry': G_dry,
        'K_min': K_min,
        'Rho_min': Rho_min,
        'K_brine': K_brine,
        'Rho_brine': Rho_brine,
        'K_co2': K_co2,
        'Rho_co2': Rho_co2,
        'phi': phi,
        'brie_component': brie_component
    }

    return rock_physics_params



# ================================================================
# ROCK PHYSICS HELPER FUNCTIONS
# ================================================================

def vrh(volumes, k, mu, rho):
    """
    Calculate the Voigt-Reuss-Hill (VRH) bounds and average properties for a mineral matrix.

    Args:
        volumes (list): List of volume fractions for each mineral.
        k (list): List of bulk moduli for each mineral.
        mu (list): List of shear moduli for each mineral.
        rho (list): List of densities for each mineral.

    Returns:
        tuple: A tuple containing the following values:
            - k_u (float): Upper bound of bulk modulus (Voigt upper bound).
            - k_l (float): Lower bound of bulk modulus (Reuss lower bound).
            - mu_u (float): Upper bound of shear modulus (Voigt upper bound).
            - mu_l (float): Lower bound of shear modulus (Reuss lower bound).
            - K_min (float): Average bulk modulus (Hill's average).
            - mu0 (float): Average shear modulus (Hill's average).
            - rho0 (float): Average density.
    """

    f = np.array(volumes)
    k = np.array(k)
    mu = np.array(mu)
    rho = np.array(rho)

    k_u = np.sum(f * k)
    k_l = 1. / np.sum(f / k)
    mu_u = np.sum(f * mu)
    mu_l = 1. / np.sum(f / mu)
    K_min = (k_u + k_l) / 2.0
    mu0 = (mu_u + mu_l) / 2.0
    rho0 = np.sum(f * rho)

    # Reuss low bound, Voigt upper bound, and Hill’s average for mineral matrix
    return k_u, k_l, mu_u, mu_l, K_min, mu0, rho0



def moduli_from_vp_vs_rho(vp, vs, rho):
    """
    Compute saturated bulk and shear moduli from seismic velocities.

    Parameters
    ----------
    vp : float or ndarray
        P-wave velocity (m/s)
    vs : float or ndarray
        S-wave velocity (m/s)
    rho : float or ndarray
        Density (kg/m³)

    Returns
    -------
    K : float or ndarray
        Saturated bulk modulus (GPa)
    G : float or ndarray
        Saturated shear modulus (GPa)
    """
    if np.any(vs <= 0):
        raise ValueError("S-wave velocity must be positive.")
    if np.any(vp <= vs):
        raise ValueError("P-wave velocity must be greater than S-wave velocity.")
    
    G = rho * vs**2 / 1e9
    K = (rho * vp**2 - 4.0 / 3.0 * rho * vs**2) / 1e9
    return K, G


def vp_vs_from_moduli_rho(K, G, rho):
    """
    Compute seismic velocities from saturated bulk and shear moduli.
    
    Parameters
    ----------
    K : float or ndarray
        Saturated bulk modulus (GPa)
    G : float or ndarray
        Saturated shear modulus (GPa)
    rho : float or ndarray
        Density (kg/m³)
        
    Returns
    -------
    vp : float or ndarray
        P-wave velocity (m/s)
    vs : float or ndarray
        S-wave velocity (m/s)
    """
    vp = np.sqrt((K * 1e9 + 4.0/3.0 * G * 1e9) / rho)
    vs = np.sqrt((G * 1e9) / rho)
    
    return vp, vs


def gassmann_sat(Kdry, Kmin, phi, Kf):
    """
    Apply Gassmann's fluid substitution to compute saturated bulk modulus.

    Parameters
    ----------
    Kdry : float or ndarray
        Dry rock bulk modulus (GPa)
    Kmin : float
        Mineral bulk modulus (GPa)
    phi : float or ndarray
        Porosity (fraction, 0-1)
    Kf : float or ndarray
        Effective fluid bulk modulus (GPa)

    Returns
    -------
    K_sat : float or ndarray
        Saturated bulk modulus (GPa)
    """
    term = phi / Kf + (1.0 - phi) / Kmin - (Kdry / (Kmin**2))
    K_sat = Kdry + ((1.0 - Kdry / Kmin)**2) / term
    
    return K_sat



# ROCK FRAME CALIBRATION
def estimate_porosity_from_density(rho_sat, rho_min, rho_fluid, clip=(0.0, 0.40)):
    """
    Estimate porosity using density mixing relation.

    Parameters
    ----------
    rho_sat : float or ndarray
        Saturated bulk density (kg/m³)
    rho_matrix : float
        Mineral grain density (kg/m³)
    rho_min : float
        Pore fluid density (kg/m³)
    clip : tuple of float
        Minimum and maximum porosity limits

    Returns
    -------
    phi : float or ndarray
        Estimated porosity (fraction)
    """
    phi = (rho_sat - rho_min) / (rho_fluid - rho_min + 1e-12)
    
    if np.any(phi < 0.0) or np.any(phi > 0.4):
        raise ValueError("Warning: Estimated porosity out of bounds [0, 0.4]. Clipping.")
        
    return np.clip(phi, clip[0], clip[1])


def calibrate_frame_from_baseline(vp, vs, rho, K_min, rho_min, K_brine, Rho_brine):
    """
    Estimate dry frame moduli and porosity from baseline elastic properties.

    Parameters
    ----------
    vp, vs, rho : float or ndarray
        Baseline P-wave, S-wave velocities (m/s) and density (kg/m³)
    K_min : float
        Mineral bulk modulus (GPa)
    rho_min : float
        Mineral grain density (kg/m³)
    K_brine : float
        Brine bulk modulus (GPa)
    Rho_brine : float
        Brine density (kg/m³)
        
    Returns
    -------
    phi : float or ndarray
        Estimated porosity
    Kdry : float or ndarray
        Dry bulk modulus (GPa)
    Gdry : float or ndarray
        Dry shear modulus (GPa)
    """

    K_sat0, G_sat0 = moduli_from_vp_vs_rho(vp, vs, rho)
    phi = estimate_porosity_from_density(rho, rho_min, Rho_brine)
    Gdry = G_sat0  # Gassmann assumption: shear unaffected by fluid

    # Golden-section search for Kdry
    Kdry_lo = np.full_like(K_sat0, 1e-3)
    Kdry_hi = np.full_like(K_sat0, K_min - 1e-3)

    def misfit(Kdry_trial):
        Kmod = gassmann_sat(Kdry_trial, K_min, phi, K_brine)
        return (Kmod - K_sat0)**2

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = Kdry_lo, Kdry_hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = misfit(c)
    fd = misfit(d)

    for _ in range(50):
        mask = fc < fd
        b = np.where(mask, d, b)
        a = np.where(mask, a, c)
        d = np.where(mask, c, d)
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = misfit(c)
        fd = misfit(d)

    # Final estimate
    Kdry = 0.5 * (a + b)
    
    return phi, Kdry, Gdry



def Brie(Kw, Kgas, Sw, e):
    """Brie empirical fluid mixing law

    Parameters
    ----------
    Kw : float
        bulk modulus of fluid phase
    Kgas : float
        bulk modulus of gas phase
    Sw : float or array
        water saturation
    e : int
        Brie component

    Returns
    -------
    float or array
        Kf: effective fluid propertie
    """        
    
    Kf= (Kw-Kgas)*Sw**e+Kgas 
    return Kf   


