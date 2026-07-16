import numpy as np
import pandas as pd
import os
import torch

import seisfwi.defaults as defaults

def to_tensor(*args):
    return [torch.tensor(arg, dtype=defaults.dtype, device=defaults.device) for arg in args]


class RockPhysicsCO2(torch.nn.Module):
    """
    A PyTorch module that calculates the modified rock properties using the Fluid Replacement Method (FRM).
    """

    def __init__(self, 
                 K_co2_post, Rho_co2_post,
                 K_brine, Rho_brine, 
                 k0, mu0, rho0, 
                 totalporo, compos1, compos2, alpha1, alpha2):

        super(RockPhysicsCO2, self).__init__()

        # convert to tensor
        K_co2_post, Rho_co2_post, K_brine, Rho_brine, k0, mu0, rho0, \
        totalporo, compos1, compos2, alpha1, alpha2 = to_tensor(K_co2_post, \
        Rho_co2_post, K_brine, Rho_brine, k0, mu0, rho0, totalporo, \
        compos1, compos2, alpha1, alpha2 )

        self.K_co2_post = K_co2_post
        self.Rho_co2_post = Rho_co2_post
        self.K_brine = K_brine
        self.Rho_brine = Rho_brine
        self.k0 = k0
        self.mu0 = mu0
        self.rho0 = rho0
        self.totalporo = totalporo
        self.compos1 = compos1
        self.compos2 = compos2
        self.alpha1 = alpha1
        self.alpha2 = alpha2



    def forward(self, SCO2):
        return RockPhysicsOperator(SCO2, 
                                   self.K_co2_post, self.Rho_co2_post,
                                   self.K_brine, self.Rho_brine, 
                                   self.k0, self.mu0, self.rho0, 
                                   self.totalporo, self.compos1, self.compos2, self.alpha1, self.alpha2)


def RockPhysicsOperator(SCO2, 
                        K_co2_post, Rho_co2_post,
                        K_brine, Rho_brine, 
                        k0, mu0, rho0, 
                        totalporo, compos1, compos2, alpha1, alpha2):
    """
    Calculate the modified rock properties using the Fluid Replacement Method (FRM).

    Args:
        SCO2 (float): The CO2 saturation.
        K_co2_post (float): The bulk modulus of CO2 in the post-injection state.
        Rho_co2_post (float): The density of CO2 in the post-injection state.
        K_brine (float): The bulk modulus of brine.
        Rho_brine (float): The density of brine.
        k0 (float): The bulk modulus of the rock matrix.
        mu0 (float): The shear modulus of the rock matrix.
        rho0 (float): The density of the rock matrix.
        totalporo (float): The total porosity of the rock.
        compos1 (float): The composition of the first mineral phase.
        compos2 (float): The composition of the second mineral phase.
        alpha1 (float): The coordination number of the first mineral phase.
        alpha2 (float): The coordination number of the second mineral phase.
    Returns:
        tuple: A tuple containing the modified rock properties (Vp_post, Vs_post, rho_post).

    """
    if not torch.is_tensor(SCO2):
        SCO2 = torch.tensor(SCO2)

    # injected fluid    
    e = 2
    SW = 1 - SCO2
    rho_inj = (SCO2 * Rho_co2_post) + (SW * Rho_brine)
    k_inj   = ((K_brine - K_co2_post) * (SW ** e)) + K_co2_post
    mu_inj  = 0
    
    # Saturated Vp, Vs, rho, use Kuster-Toksoz
    stuffs1_inj = stuffs_torch(k0, k_inj, mu0, mu_inj, totalporo, compos1, alpha1)
    stuffs2_inj = stuffs_torch(k0, k_inj, mu0, mu_inj, totalporo, compos2, alpha2)
    
    ci_1_inj = stuffs1_inj[4]
    ci_2_inj = stuffs2_inj[4]
    
    PQ_1_inj = PQ_torch(stuffs1_inj[0], stuffs1_inj[1], stuffs1_inj[2], stuffs1_inj[5], stuffs1_inj[6])
    PQ_2_inj = PQ_torch(stuffs2_inj[0], stuffs2_inj[1], stuffs2_inj[2], stuffs2_inj[5], stuffs2_inj[6])
    
    sigma_P_inj = (ci_1_inj * PQ_1_inj[0]) + (ci_2_inj * PQ_2_inj[0])
    sigma_Q_inj = (ci_1_inj * PQ_1_inj[1]) + (ci_2_inj * PQ_2_inj[1])
    
    KusTok_inj = KusterToksoz_torch(sigma_P_inj, sigma_Q_inj, k0, mu0, k_inj, rho0, rho_inj)
    rho_post = KusTok_inj[2]
    Vp_post = KusTok_inj[3] * 1000
    Vs_post = KusTok_inj[4] * 1000
    
    # remove background: Caveat: this is only valid for CO2 where max is the background for vp and rho, and min is the background for vs
    Vp_post = Vp_post - torch.max(Vp_post) 
    Vs_post = Vs_post - torch.min(Vs_post)
    rho_post = rho_post - torch.max(rho_post)
    
    return Vp_post, Vs_post, rho_post



class RockPhysicsGassmann(torch.nn.Module):
    """
    A PyTorch module that calculates the modified rock properties using the Fluid Replacement Method (FRM).
    
    K_dry: Dry bulk modulus
    G_dry: Dry shear modulus
    K_min: Mineral bulk modulus
    Rho_min: Mineral density
    K_brine: Brine bulk modulus
    Rho_brine: Brine density
    K_co2: CO2 bulk modulus
    Rho_co2: CO2 density
    phi: Porosity
    brie_component: Component for Brie mixing model (default is 2 for patchy saturation)
    """

    def __init__(self, vp_res, vs_res, rho_res, K_dry, G_dry, K_min, Rho_min, K_brine, Rho_brine, 
                 K_co2, Rho_co2, phi, brie_component=2):
        
        super(RockPhysicsGassmann, self).__init__()

        # convert to tensor
        vp_res, vs_res, rho_res, K_dry, G_dry, K_min, Rho_min, K_brine, Rho_brine, K_co2, Rho_co2, phi, brie_component = to_tensor(vp_res, vs_res, rho_res, K_dry, G_dry, K_min, Rho_min, K_brine, Rho_brine, K_co2, Rho_co2, phi, brie_component)

        self.vp_res = vp_res
        self.vs_res = vs_res
        self.rho_res = rho_res
        self.K_dry = K_dry
        self.G_dry = G_dry
        self.K_min = K_min
        self.Rho_min = Rho_min
        self.K_brine = K_brine
        self.Rho_brine = Rho_brine
        self.K_co2 = K_co2
        self.Rho_co2 = Rho_co2
        self.phi = phi
        self.brie_component = brie_component

    def forward(self, SCO2):
        vp_sat, vs_sat, rho_sat = RockPhysicsGassmannOperator(SCO2, 
                                   self.K_dry, self.G_dry,
                                   self.K_min, self.Rho_min, 
                                   self.K_brine, self.Rho_brine, 
                                   self.K_co2, self.Rho_co2, 
                                   self.phi, self.brie_component)
        
        return vp_sat - self.vp_res, vs_sat - self.vs_res, rho_sat - self.rho_res


def RockPhysicsGassmannOperator(Sco2, 
                                K_dry, G_dry, 
                                K_min, Rho_min, 
                                K_brine, Rho_brine, 
                                K_co2, Rho_co2, 
                                phi, brie_component=2):
    """
    Calculate the modified rock properties using the Fluid Replacement Method (FRM).

    """
    
    if not torch.is_tensor(Sco2):
        Sco2 = torch.tensor(Sco2, dtype=defaults.dtype, device=defaults.device)

    # Mix fluid with patchy saturation: Kf = (Kw-Kgas)*Sw**e+Kgas 
    K_fluid  = Brie(K_brine, K_co2, 1-Sco2, brie_component)
    Rho_fluid = (1 - Sco2) * Rho_brine + Sco2 * Rho_co2

    # Apply Gassmann's fluid substitution to compute saturated bulk modulus.
    Ksat = gassmann_sat(K_dry, K_min, phi, K_fluid)

    # update the elastic properties, shear unaffected by fluid substitution
    rho_sat = (1.0 - phi) * Rho_min + phi * Rho_fluid
    Gsat = G_dry
    vp_sat, vs_sat = vp_vs_from_moduli_rho(Ksat, Gsat, rho_sat)

    return vp_sat, vs_sat, rho_sat



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
            - k0 (float): Average bulk modulus (Hill's average).
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
    k0 = (k_u + k_l) / 2.0
    mu0 = (mu_u + mu_l) / 2.0
    rho0 = np.sum(f * rho)

    # Reuss low bound, Voigt upper bound, and Hill’s average for mineral matrix

    return k_u, k_l, mu_u, mu_l, k0, mu0, rho0


def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
    """
    Calculate the modified rock properties using the Fluid Replacement Method (FRM).

    Args:
        vp1 (float): P-wave velocity of the original rock (in m/s).
        vs1 (float): S-wave velocity of the original rock (in m/s).
        rho1 (float): Density of the original rock (in g/cm^3).
        rho_f1 (float): Density of the fluid in the original rock (in g/cm^3).
        k_f1 (float): Bulk modulus of the fluid in the original rock (in GPa).
        rho_f2 (float): Density of the fluid in the replacement rock (in g/cm^3).
        k_f2 (float): Bulk modulus of the fluid in the replacement rock (in GPa).
        k0 (float): Bulk modulus of the mineral matrix (in GPa).
        phi (float): Porosity of the rock (dimensionless).

    Returns:
        tuple: A tuple containing the modified rock properties:
            - vp2 (float): P-wave velocity of the replacement rock (in m/s).
            - vs2 (float): S-wave velocity of the replacement rock (in m/s).
            - rho2 (float): Density of the replacement rock (in g/cm^3).
            - k_s2 (float): Bulk modulus of the replacement rock (in GPa).
    """
    vp1  = vp1 / 1000.
    vs1  = vs1 / 1000.
    mu1  = rho1 * vs1**2.
    k_s1 = rho1 * vp1**2 - (4./3.)*mu1
    
    # The dry rock bulk modulus
    K_dry = (k_s1 * ((phi*k0)/k_f1+1-phi)-k0) / ((phi*k0)/k_f1+(k_s1/k0)-1-phi)

    # Now we can apply Gassmann to get the new values
    k_s2 = K_dry + (1- (K_dry/k0))**2 / ( (phi/k_f2) + ((1-phi)/k0) - (K_dry/k0**2) )
    rho2 = rho1-phi * rho_f1+phi * rho_f2
    mu2  = mu1
    vp2  = np.sqrt(((k_s2+(4./3)*mu2))/rho2)
    vs2  = np.sqrt((mu2/rho2))

    return vp2*1000, vs2*1000, rho2, k_s2



# https://github.com/yohanesnuwara/carbon-capture-and-storage/tree/master/lib
def stuffs_torch(Km, Kf, Gm, Gf, totalporo, compos, alpha):
    A = Gf / Gm - 1.0
    B = (Kf / Km - Gf / Gm) / 3.0
    R = Gm / (Km + (4.0 / 3.0) * Gm)
    Fm = (Gm / 6.0) * (9.0 * Km + 8.0 * Gm) / (Km + 2.0 * Gm) #zeta
    ci = compos*totalporo
    theta = alpha * (torch.acos(alpha) - alpha * torch.sqrt(1.0 - alpha * alpha)) / (1.0 - alpha * alpha) ** (3.0 / 2.0)
    f = alpha * alpha * (3.0 * theta - 2.0) / (1.0 - alpha * alpha)

    return (A, B, R, Fm, ci, theta, f)

def PQ_torch(A, B, R, theta, f):
    F1 = 1.0 + A * (1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta - 4.0 / 3.0))
    F2 = 1.0 + A * (1.0 + 1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta)) + B * (3.0 - 4.0 * R) + A * (A + 3.0 * B) * (1.5 - 2.0 * R) * (f + theta - R * (f - theta + 2.0 * theta * theta))
    F3 = 1.0 + A * (1.0 - f - 1.5 * theta + R * (f + theta))
    F4 = 1.0 + (A / 4.0) * (f + 3.0 * theta - R * (f - theta))
    F5 = A * (-f + R * (f + theta - 4.0 / 3.0)) + B * theta * (3.0 - 4.0 * R)
    F6 = 1.0 + A * (1.0 + f - R * (f + theta)) + B * (1.0 - theta) * (3.0 - 4.0 * R)
    F7 = 2.0 + (A / 4.0) * (3.0 * f + 9.0 * theta - R * (3.0 * f + 5.0 * theta)) + B * theta * (3.0 - 4.0 * R)
    F8 = A * (1.0 - 2.0 * R + (f / 2.0) * (R - 1.0) + (theta / 2.0) * (5.0 * R - 3.0)) + B * (1.0 - theta) * (3.0 - 4.0 * R)
    F9 = A * ((R - 1.0) * f - R * theta) + B * theta * (3.0 - 4.0 * R)
    P = 3.0 * F1 / F2
    Q = 2.0 / F3 + 1.0 / F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)
    
    return (P,Q)

def KusterToksoz_torch(sigma_P, sigma_Q, Km, Gm, Kf, rhom, rhof):
    K_sat = ((3 * Km * (3 * Km + 4 * Gm)) + (4 * Gm * (Kf - Km) * sigma_P)) / ((3 * (3 * Km + 4 * Gm)) - (3 * (Kf - Km) * sigma_P))
    G_sat = ((25 * (Gm**2) * (3 * Km + 4 * Gm)) - ((Gm**2) * (9 * Km + 8 * Gm) * sigma_Q)) / ((25 * Gm * (3 * Km + 4 * Gm)) + (6 * Gm * (Km + 2 * Gm) * sigma_Q))
    rho_sat = (rhom*(1-0.14))+(rhof*0.14)
    Vp_sat = torch.sqrt((K_sat + 4/3*G_sat) / rho_sat)
    Vs_sat = torch.sqrt(G_sat / rho_sat)
    
    return (K_sat, G_sat, rho_sat, Vp_sat, Vs_sat)


"""
Fluid Inclusion Modelling with Kuster-Toksoz Method (simplification of Wang, 1997)
By: Yohanes Nuwara
"""

def stuffs(Km, Kf, Gm, Gf, totalporo, compos, alpha):
    A = Gf / Gm - 1.0
    B = (Kf / Km - Gf / Gm) / 3.0
    R = Gm / (Km + (4.0 / 3.0) * Gm)
    Fm = (Gm / 6.0) * (9.0 * Km + 8.0 * Gm) / (Km + 2.0 * Gm) #zeta
    ci = compos*totalporo
    theta = alpha * (np.arccos(alpha) - alpha * np.sqrt(1.0 - alpha * alpha)) / (1.0 - alpha * alpha) ** (3.0 / 2.0)
    f = alpha * alpha * (3.0 * theta - 2.0) / (1.0 - alpha * alpha)
    return (A, B, R, Fm, ci, theta, f)

def PQ(A, B, R, theta, f):
    F1 = 1.0 + A * (1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta - 4.0 / 3.0))
    F2 = 1.0 + A * (1.0 + 1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta)) + B * (3.0 - 4.0 * R) + A * (A + 3.0 * B) * (
        1.5 - 2.0 * R) * (f + theta - R * (f - theta + 2.0 * theta * theta))
    F3 = 1.0 + A * (1.0 - f - 1.5 * theta + R * (f + theta))
    F4 = 1.0 + (A / 4.0) * (f + 3.0 * theta - R * (f - theta))
    F5 = A * (-f + R * (f + theta - 4.0 / 3.0)) + B * theta * (3.0 - 4.0 * R)
    F6 = 1.0 + A * (1.0 + f - R * (f + theta)) + B * (1.0 - theta) * (3.0 - 4.0 * R)
    F7 = 2.0 + (A / 4.0) * (3.0 * f + 9.0 * theta - R * (3.0 * f + 5.0 * theta)) + B * theta * (3.0 - 4.0 * R)
    F8 = A * (1.0 - 2.0 * R + (f / 2.0) * (R - 1.0) + (theta / 2.0) * (5.0 * R - 3.0)) + B * (1.0 - theta) * (
        3.0 - 4.0 * R)
    F9 = A * ((R - 1.0) * f - R * theta) + B * theta * (3.0 - 4.0 * R)
    P = 3.0 * F1 / F2
    Q = 2.0 / F3 + 1.0 / F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)
    return(P,Q)

def KusterToksoz(sigma_P, sigma_Q, Km, Gm, Kf, rhom, rhof):
    K_sat = ((3 * Km * (3 * Km + 4 * Gm)) + (4 * Gm * (Kf - Km) * sigma_P)) / ((3 * (3 * Km + 4 * Gm)) -
        (3 * (Kf - Km) * sigma_P))
    G_sat = ((25 * (Gm**2) * (3 * Km + 4 * Gm)) - ((Gm**2) * (9 * Km + 8 * Gm) * sigma_Q)) / ((25 * Gm * (3 * Km + 4 * Gm))
        + (6 * Gm * (Km + 2 * Gm) * sigma_Q))
    rho_sat = (rhom*(1-0.14))+(rhof*0.14)
    Vp_sat = np.sqrt((K_sat + 4/3*G_sat) / rho_sat)
    Vs_sat = np.sqrt(G_sat / rho_sat)
    return (K_sat, G_sat, rho_sat, Vp_sat, Vs_sat)


"""
Fluid Property Modelling with Batzle-Wang (1991) for brine and CO2 (gas-assumed)
By: Yohanes Nuwara
"""

#Batzle-Wang for brine
def BW_brine_density(temp, Pp_baseline, salinity):
    rhow = 1+(0.000001)*((-80*temp)-(3.3*(temp**2))+(0.00175*(temp**3))+(489*Pp_baseline)-(2*temp*Pp_baseline)+
            (0.016*(temp**2)*Pp_baseline)-((0.000013)*(temp**3)*Pp_baseline)-(0.333*(Pp_baseline**2)-(0.002*temp*
            (Pp_baseline**2))))
    rhobrine = rhow+(0.668*salinity)+(0.44*salinity**2)+((10E-6)*salinity*((300*Pp_baseline)-(2400*Pp_baseline*salinity)+
            (temp*(80+(3*temp)-(3300*salinity)-(13*Pp_baseline)+(47*Pp_baseline*salinity)))))
    return(rhobrine)

def BW_brine_bulk(temp, Pp_baseline, salinity, rhobrine):
    vw1 = 1402.85*(temp**0)*(Pp_baseline**0)
    vw2 = 4.871*(temp**1)*(Pp_baseline**0)
    vw3 = -0.04783*(temp**2)*(Pp_baseline**0)
    vw4 = (1.487E-04)*(temp**3)*(Pp_baseline**0)
    vw5 = (-2.197E-07)*(temp**4)*(Pp_baseline**0)
    vw6 = 1.524*(temp**0)*(Pp_baseline**1)
    vw7 = -0.0111*(temp**1)*(Pp_baseline**1)
    vw8 = (2.747E-04)*(temp**2)*(Pp_baseline**1)
    vw9 = (-6.503E-07)*(temp**3)*(Pp_baseline*1)
    vw10 = (7.987E-10)*(temp**4)*(Pp_baseline**1)
    vw11 = (3.437E-03)*(temp**0)*(Pp_baseline**2)
    vw12 = (1.739E-04)*(temp**1)*(Pp_baseline**2)
    vw13 = (-2.135E-06)*(temp**2)*(Pp_baseline**2)
    vw14 = (-1.455E-08)*(temp**3)*(Pp_baseline**2)
    vw15 = (5.23E-11)*(temp**4)*(Pp_baseline**2)
    vw16 = (-1.197E-05)*(temp**0)*(Pp_baseline**3)
    vw17 = (-1.628E-06)*(temp**1)*(Pp_baseline**3)
    vw18 = (1.237E-08)*(temp**2)*(Pp_baseline**3)
    vw19 = (1.327E-10)*(temp**3)*(Pp_baseline**3)
    vw20 = (-4.614E-13)*(temp**4)*(Pp_baseline**3)
    vw = [vw1, vw2, vw3, vw4, vw5, vw6, vw7, vw8, vw9, vw10, vw11, vw12, vw13, vw14, vw15,
          vw16, vw17, vw18, vw19, vw20]
    vwsum = sum(vw)
    vbrine = vwsum+(salinity*(1170-(9.6*temp)+(0.055*(temp**2))-((8.5E-05)*(temp**3))+(2.6*Pp_baseline)-(0.0029*temp*Pp_baseline)-
        (0.0476*(Pp_baseline**2))))+((salinity**1.5)*(780-(10*Pp_baseline)+(0.16*(Pp_baseline**2))))-(1820*(salinity**2))
    Kbrine = rhobrine*(vbrine**2)*(0.000001) #in GPa
    return(Kbrine)

def BW_gas_density(temp, SG, Pp_post):
    R = 8.314472 #gas constant
    temp_abs = temp + 273.15 #kelvin
    P_pr = Pp_post/(4.892-(0.40486*SG))
    T_pr = temp_abs/(94.72+(170.75*SG))
    E = 0.109*((3.85-T_pr)**2)*np.exp(-1*(0.45+(8*((0.56-(1/T_pr))**2)))*((P_pr**1.2)/T_pr))
    Z = ((0.03+(0.00527*((3.5-T_pr)**3)))*P_pr)+((0.642*T_pr)-(0.007*(T_pr**4))-0.52)+E
    rhogas = (28.8 * SG * Pp_post) / (Z * R * temp_abs)
    return(rhogas)

def BW_gas_bulk(temp, SG, Pp_post):
    temp_abs = temp + 273.15  # kelvin
    P_pr = Pp_post / (4.892 - (0.40486 * SG))
    T_pr = temp_abs / (94.72 + (170.75 * SG))
    E = 0.109*((3.85-T_pr)**2)*np.exp(-1*(0.45+(8*((0.56-(1/T_pr))**2)))*((P_pr**1.2)/T_pr))
    Z = ((0.03+(0.00527*((3.5-T_pr)**3)))*P_pr)+((0.642*T_pr)-(0.007*(T_pr**4))-0.52)+E
    Gamma0 = 0.85 + (5.6 / (P_pr + 2)) + (27.1 / (P_pr + 3.5) ** 2) - (8.7 * np.exp(-1 * 0.65 * (P_pr + 1)))
    F = -1.2*((P_pr**0.2)/T_pr)*(0.45+(8*(0.56-(1/T_pr))**2))*np.exp(-1*(0.45+(8*(0.56-(1/T_pr)**2)))*(((P_pr)**1.2)/T_pr))
    doZ_doPpr = 0.03 + 0.00527 * ((3.5 - T_pr) ** 3) + ((0.109 * (3.85 - T_pr) ** 2) * F)
    Kgas = (Pp_post * Gamma0 / (1 - ((P_pr / Z) * doZ_doPpr))) / 1000 #in GPa
    return(Kgas)


"""
CO2 Acoustic Properties from Equation of State database Span and Wagner (1996)
By: Yohanes Nuwara
"""




def EOS(Pp_post, temp):
    """ Calculate CO2 density, bulk modulus, and sound velocity from EOS database
    
    Pp_post : pressure in MPa
    temp : temperature in deg C
    """

    # path of this file
    path = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(path, 'eostable.npy')

    # load EOS database in a form of npy file
    try:
        load_eos = np.load(file)
    except:
        print('EOS database not found')
        return

    load_eos = pd.DataFrame(load_eos, columns=['Temperature (deg c)', 
                                               'Pressure (Pa)', 
                                               'Density (g/cc)',
                                               'Sound speed (m/s)'])

    # convert temperature to kelvin
    temp = temp + 273
    # normalize pressure
    Pp_rounded = round(Pp_post/5)*5 #round to nearest 5 multiples, e.g. 32 to 30, 34 to 35, 36 to 35, 38 to 50
    #normalize temp
    temp_rounded = round(temp/1)*1 #round to nearest number, e.g. 273.2 to 273, and 608.6 to 609
    #find the normalized temp input in the dataframe
    findtemp = load_eos.loc[load_eos['Temperature (deg c)'] == temp_rounded]

    # code for interpolation
    if Pp_post < Pp_rounded:
        Pp_rounded_low = (Pp_rounded - 5) * 1E+06
        Pp_rounded = Pp_rounded * 1E+06
        Pp_post = Pp_post * 1E+06

        #find density data from dataframe
        findPp_low = findtemp.loc[findtemp['Pressure (Pa)'] == Pp_rounded_low]
        findPp_rounded = findtemp.loc[findtemp['Pressure (Pa)'] == Pp_rounded]

        #for rho CO2
        rho_rounded_low = findPp_low.iloc[0]['Density (g/cc)']
        rho_rounded = findPp_rounded.iloc[0]['Density (g/cc)']
        rhoCO2 = (((Pp_post - Pp_rounded_low)*(rho_rounded - rho_rounded_low)) /
                  (Pp_rounded - Pp_rounded_low)) + rho_rounded_low

        #for bulk CO2
        vel_rounded_low = findPp_low.iloc[0]['Sound speed (m/s)']
        vel_rounded = findPp_rounded.iloc[0]['Sound speed (m/s)']
        velCO2 = (((Pp_post - Pp_rounded_low)*(vel_rounded - vel_rounded_low)) /
                  (Pp_rounded - Pp_rounded_low)) + vel_rounded_low
        KCO2 = (rhoCO2*(velCO2**2)) / 1E+06

    elif Pp_post > Pp_rounded:
        Pp_rounded_high = (Pp_rounded + 5) * 1E+06
        Pp_rounded = Pp_rounded * 1E+06
        Pp_post = Pp_post * 1E+06

        #find density data from dataframe
        findPp_high = findtemp.loc[findtemp['Pressure (Pa)'] == Pp_rounded_high]
        findPp_rounded = findtemp.loc[findtemp['Pressure (Pa)'] == Pp_rounded]

        # for rho CO2
        rho_rounded_high = findPp_high.iloc[0]['Density (g/cc)']
        rho_rounded = findPp_rounded.iloc[0]['Density (g/cc)']
        rhoCO2 = (((Pp_post - Pp_rounded) * (rho_rounded_high - rho_rounded)) /
                  (Pp_rounded_high - Pp_rounded)) + rho_rounded

        # for bulk CO2
        vel_rounded_high = findPp_high.iloc[0]['Sound speed (m/s)']
        vel_rounded = findPp_rounded.iloc[0]['Sound speed (m/s)']
        velCO2 = (((Pp_post - Pp_rounded) * (vel_rounded_high - vel_rounded)) /
                  (Pp_rounded_high - Pp_rounded)) + vel_rounded
        KCO2 = (rhoCO2*(velCO2**2)) / 1E+06

    elif Pp_post == Pp_rounded:
        Pp_post = Pp_post * 1E+06

        #find density data from dataframe
        findPp_exact = findtemp.loc[findtemp['Pressure (Pa)'] == Pp_post]

        #for rho CO2
        rho_exact = findPp_exact.iloc[0]['Density (g/cc)']
        rhoCO2 = rho_exact

        # for bulk CO2
        vel_exact = findPp_exact.iloc[0]['Sound speed (m/s)']
        velCO2 = vel_exact
        KCO2 = (rhoCO2 * (velCO2 ** 2)) / 1E+06

    return rhoCO2, KCO2, velCO2



def vp_vs_from_moduli_rho(K, G, rho):
    """
    Compute seismic velocities from saturated bulk and shear moduli.
    
    Parameters
    ----------
    K : float or ndarray / torch.Tensor
        Saturated bulk modulus (GPa)
    G : float or ndarray / torch.Tensor
        Saturated shear modulus (GPa)
    rho : float or ndarray / torch.Tensor
        Density (kg/m³)
        
    Returns
    -------
    vp : same type as input
        P-wave velocity (m/s)
    vs : same type as input
        S-wave velocity (m/s)
    """
    # Convert GPa to Pa
    K_pa = K * 1e9
    G_pa = G * 1e9

    vp = ((K_pa + 4.0 / 3.0 * G_pa) / rho) ** 0.5
    vs = (G_pa / rho) ** 0.5
    
    return vp, vs


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


def gassmann_sat(K_dry, K_min, phi, Kf):
    """
    Apply Gassmann's fluid substitution to compute saturated bulk modulus.

    Parameters
    ----------
    K_dry : float or ndarray
        Dry rock bulk modulus (GPa)
    K_min : float
        Mineral bulk modulus (GPa)
    phi : float or ndarray
        Porosity (fraction, 0-1)
    Kf : float or ndarray
        Effective fluid bulk modulus (GPa)

    Returns
    -------
    Ksat : float or ndarray
        Saturated bulk modulus (GPa)
    """
    term = phi / Kf + (1.0 - phi) / K_min - (K_dry / (K_min**2))
    Ksat = K_dry + ((1.0 - K_dry / K_min)**2) / term
    
    return Ksat