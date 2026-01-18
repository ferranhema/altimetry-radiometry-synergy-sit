#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
This script computes the complex dielectric constant of sea ice by considering
brine inclusions as ellipsoids. It includes three formulations:
1. Maxwell-Garnett (v=0)
2. Coherent Potential (v=1)
3. Polder van Santen (v=2)

It is based on the paper:
"Particle shape effects on the effective permittivity of anisotropic or isotropic
media consisting of aligned or randomly oriented ellipsoidal particles."
(Scott B. Jones and Shmulik P. Friedman, 2000)
"""
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
import numpy as np
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
def depolarization_factor(aspect_ratio, c):
    """
    Computes the depolarization factors Na, Nb, Nc based on the aspect ratio.
    """
    Na = ((1) / (1 + 1.6 * (aspect_ratio) + 0.4 * (aspect_ratio)**2))
    Nb = 0.5 * (1 - Na)
    Nc = Nb
    return Na, Nb, Nc

def e_ice_brine_vi(freq, sice, tice):
    """
    Calculates the brine volume (vi) and the permittivity of pure ice and brine.
    
    Args:
        freq: Frequency in Hz.
        sice: Sea ice salinity.
        tice: Sea ice temperature (Celsius).
        
    Returns:
        tuple: (vi, e_ice, e_brine)
    """
    # Polynomial coefficients for Brine Salinity (Sb)
    if tice >= -8.2:
        a, b, c, d = 1.725, -18.756, -0.3964, 0
    elif tice >= -22.9:
        a, b, c, d = 57.041, -9.929, -0.16204, -0.002396
    elif tice >= -36.8:
        a, b, c, d = 242.94, 1.5299, 0.0429, 0
    elif tice >= -43.2:
        a, b, c, d = 508.18, 14.535, 0.2018, 0
    else:
        a, b, c, d = 508.18, 14.535, 0.2018, 0

    Sb = a + b * tice + c * tice**2 + d * tice**3
    Nb = 1.707e-2 * Sb + 1.205e-5 * Sb**2 + 4.058e-9 * Sb**3
    Nb = Nb * 0.9141  # Correction for sea water
    
    # --- Brine Permittivity (Stogryn and Desargant, 1985) ---
    epsiwoo = 4.9
    e0_const = 8.854e-12
    
    eps = 88.045 - 0.4147 * tice + 6.295e-4 * (tice**2) + 1.075e-5 * (tice**3)
    a_coeff = 1 - 0.255 * Nb + 5.15e-2 * (Nb**2) - 6.89e-3 * (Nb**3)
    eb0 = eps * a_coeff
    
    rel = 1.1109e-10 - 3.824e-12 * tice + 6.938e-14 * (tice**2) - 5.096e-16 * (tice**3)
    b_coeff = 1 + 0.146e-2 * tice * Nb - 4.89e-2 * Nb - 2.97e-2 * (Nb**2) + 5.64e-3 * (Nb**3)
    relax = rel * b_coeff
    
    D_val = 25 - tice
    sig = Nb * (10.39 - 2.378 * Nb + 0.683 * (Nb**2) - 0.135 * (Nb**3) + 1.01e-2 * (Nb**4))
    c_coeff = 1 - 1.96e-2 * D_val + 8.08e-5 * (D_val**2) - Nb * D_val * (
        3.02e-5 + 3.92e-5 * D_val + Nb * (1.72e-5 - 6.58e-6 * D_val)
    )
    conb = c_coeff * sig
    
    omega_tau = relax * freq
    denom = 1 + (omega_tau**2)
    
    eb_real = epsiwoo + ((eb0 - epsiwoo) / denom)
    eb_imag = ((omega_tau * (eb0 - epsiwoo)) / denom) + (conb / (2 * 3.14159 * freq * e0_const))
    e_brine = complex(eb_real, eb_imag)
    
    # --- Brine Volume Fraction (Cox and Weeks, 1983) ---
    rho_ice = 0.917 - 0.1404e-3 * tice  # Mg/m^3
    
    if tice > -2:
        a1, b1, c1, d1 = -0.041221, -18.407, 0.58402, 0.21454
        a2, b2, c2, d2 = 0.090312, -0.016111, 1.2291e-4, 1.3603e-4
    elif tice >= -22.9:
        a1, b1, c1, d1 = -4.732, -22.45, -0.6397, -0.01074
        a2, b2, c2, d2 = 0.08903, -0.01763, -5.330e-4, -8.801e-6
    else:
        a1, b1, c1, d1 = 9899, 1309, 55.27, 0.7160
        a2, b2, c2, d2 = 8.547, 1.089, 0.04518, 5.819e-4
                
    F1 = a1 + b1 * tice + c1 * (tice**2) + d1 * (tice**3)
    F2 = a2 + b2 * tice + c2 * (tice**2) + d2 * (tice**3)
    vi = (rho_ice * sice) / (F1 - rho_ice * sice * F2)
    
    # --- Pure Ice Permittivity (Shokr 1998) ---
    epi = 3.1884 + 9.1e-4 * tice
    theta = 300 / (tice + 273.15) - 1
    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    beta = ((0.502 - 0.131 * theta) / (1 + theta)) * 1e-4 + (0.542e-6 * ((1 + theta) / (theta + 0.0073))**2)
    
    epii = (alpha / (freq / 1e9)) + (beta * (freq / 1e9))
    e_ice = complex(epi, epii)
    
    return vi, e_ice, e_brine

def e_eff_i(e_eff, v, Ni, sice, tice):
    """
    Helper term calculation for Isotropic mixing (MG/CP).
    """
    vi, e0, e1 = e_ice_brine_vi(1.4e9, sice, tice)
    # Note: e0 = pure ice, e1 = brine
    
    denom = 3 * (e0 + v * (e_eff - e0) + Ni * (e1 - e0))
    e_eff_i1 = (vi * (e1 - e0) * (e0 + v * (e_eff - e0))) / denom
    e_eff_i2 = (vi * Ni * (e1 - e0)) / denom
    
    return e_eff_i1, e_eff_i2

def e_eff_ps(e_eff, Na, Nb, Nc, sice, tice):
    """
    Helper term calculation for Polder van Santen mixing.
    """
    vi, e0, e1 = e_ice_brine_vi(1.4e9, sice, tice)
    
    va = 1 - Na
    vb = 1 - Nb
    vc = 1 - Nc
    
    denom_a = 3 * (e0 + va * (e_eff - e0) + Na * (e1 - e0))
    e_eff_a1 = (vi * (e1 - e0) * (e0 + va * (e_eff - e0))) / denom_a
    e_eff_a2 = (vi * Na * (e1 - e0)) / denom_a
    
    denom_b = 3 * (e0 + vb * (e_eff - e0) + Nb * (e1 - e0))
    e_eff_b1 = (vi * (e1 - e0) * (e0 + vb * (e_eff - e0))) / denom_b
    e_eff_b2 = (vi * Nb * (e1 - e0)) / denom_b
    
    denom_c = 3 * (e0 + vc * (e_eff - e0) + Nc * (e1 - e0))
    e_eff_c1 = (vi * (e1 - e0) * (e0 + vc * (e_eff - e0))) / denom_c
    e_eff_c2 = (vi * Nc * (e1 - e0)) / denom_c
    
    return e_eff_a1, e_eff_a2, e_eff_b1, e_eff_b2, e_eff_c1, e_eff_c2

def e_eff_mix(axis_ratio, c, v, sice, tice):
    """
    Main function to compute complex permittivity.
    
    Args:
        axis_ratio (float): Aspect ratio of the ellipsoids.
        c (float): Unused parameter.
        v (int): Mixing model selector. 
                 0 = Maxwell-Garnett
                 1 = Coherent Potential
                 2 = Polder van Santen
        sice (float): Salinity.
        tice (float): Temperature.
    """
    # Initialize logic
    Na, Nb, Nc = depolarization_factor(axis_ratio, c)
    vi, e_eff0, e_brine = e_ice_brine_vi(1.4e9, sice, tice)
    
    e0 = e_eff0  # Initial guess
    m = 0
    error_real = 1.2
    error_imag = 1.2
    
    if v == 0 or v == 1: 
        # Maxwell-Garnett / Coherent potential
        while (error_real > 0.001 or error_imag > 0.001):
            e_eff_a1, e_eff_a2 = e_eff_i(e_eff0, v, Na, sice, tice)
            e_eff_b1, e_eff_b2 = e_eff_i(e_eff0, v, Nb, sice, tice)
            e_eff_c1, e_eff_c2 = e_eff_i(e_eff0, v, Nc, sice, tice)
            
            sum_terms_1 = e_eff_a1 + e_eff_b1 + e_eff_c1
            sum_terms_2 = e_eff_a2 + e_eff_b2 + e_eff_c2
            
            est = e0 + (sum_terms_1 * (1 - sum_terms_2)**-1)
            
            error_real = np.abs(np.real(e_eff0) - np.real(est))
            error_imag = np.abs(np.imag(e_eff0) - np.imag(est))
            e_eff0 = est
            
            m += 1
            if m > 30:
                break
                
        return np.real(e_eff0), np.imag(e_eff0)
    
    else: 
        # Polder van Santen (v == 2)
        while (error_real > 0.001 or error_imag > 0.001):
            (e_eff_a1, e_eff_a2, 
             e_eff_b1, e_eff_b2, 
             e_eff_c1, e_eff_c2) = e_eff_ps(e_eff0, Na, Nb, Nc, sice, tice)
            
            sum_terms_1 = e_eff_a1 + e_eff_b1 + e_eff_c1
            sum_terms_2 = e_eff_a2 + e_eff_b2 + e_eff_c2
            
            est = e0 + (sum_terms_1 * (1 - sum_terms_2)**-1)
            
            error_real = np.abs(np.real(e_eff0) - np.real(est))
            error_imag = np.abs(np.imag(e_eff0) - np.imag(est))
            e_eff0 = est
            
            m += 1
            if vi < 0.1:
                break
            if m > 30:
                break
                
        return np.real(e_eff0), np.imag(e_eff0)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#