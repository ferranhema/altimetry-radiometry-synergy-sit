#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Permittivity models for snow and sea water.

This library contains empirical models to compute the complex permittivity 
of snow and sea water based on frequency, temperature, and physical properties.
"""
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
import numpy as np
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
def epsilon_snow(freq_hz, rho_snow, temp_celsius):
    """
    Computes the complex permittivity of snow.

    Args:
        freq_hz (float): Frequency in Hz (e.g., 1.4e9 for L-band).
        rho_snow (float): Snow density in kg/m^3.
        temp_celsius (float): Snow temperature in Celsius.

    Returns:
        tuple: (EpRe, EpIm)
        - EpRe: Real part of permittivity.
        - EpIm: Imaginary part of permittivity.
    """
    # Convert inputs to units used by the empirical formula
    temp_k = temp_celsius + 273.15
    freq_ghz = freq_hz / 1e9

    # --- Real Part (EpRe) ---
    # Density-dependent mixing
    if rho_snow <= 400:
        rho_scaled = rho_snow / 1e3
        ep_re = 1 + 1.599 * rho_scaled + 1.861 * rho_scaled**3
    else:
        # Mixing formula for dense snow/firn
        vol_frac = rho_snow / 917.0
        e_host = 1.0     # Air
        e_solid = 3.215  # Pure Ice
        ep_re = (1 - vol_frac) * e_host + vol_frac * e_solid

    # --- Imaginary Part (EpIm) ---
    # Coefficients for loss calculation
    B1 = 0.0207
    b_coeff = 335
    B2 = 1.16e-11
    
    # Beta term calculation
    term_1 = (B1 / temp_k) * ( (np.exp(b_coeff / temp_k) / np.exp(b_coeff / temp_k)) - 1 )**2
    term_2 = B2 * freq_ghz**2
    term_3 = np.exp(-9.963 + 0.0372 * (temp_k - 273.16))
    
    beta = term_1 + term_2 + term_3

    # Alpha term calculation
    theta = (300.0 / temp_k) - 1
    alpha = (0.00504 + 0.0062) * np.exp(-22.1 * theta) # Note: 0.0062 usually associated with theta in models
    
    e_i = (alpha / freq_ghz) + beta * freq_ghz    
    ep_im = e_i * (0.52 * rho_snow + 0.62 * rho_snow**2)
    
    return ep_re, ep_im

def epsilon_water(freq_hz, temp_celsius, salinity_psu):
    """
    Computes the complex permittivity of sea water (Double-Debye / Stogryn-like model).

    Args:
        freq_hz (float): Frequency in Hz.
        temp_celsius (float): Temperature in Celsius.
        salinity_psu (float): Salinity in PSU.

    Returns:
        tuple: (EpRe, EpIm)
    """
    # Physical Constants
    EPSILON_0 = 8.854e-12
    
    # --- Static Permittivity (Esw0) ---
    esw_inf = 4.9
    
    # Polynomial for pure water static perm
    esw_00 = (87.174 
              - 1.949e-1 * temp_celsius 
              - 1.279e-2 * temp_celsius**2 
              + 2.491e-4 * temp_celsius**3)
    
    # Salinity correction factor 'a'
    a_coeff = (1.0 
               + 1.613e-5 * temp_celsius * salinity_psu 
               - 3.656e-3 * salinity_psu 
               + 3.21e-5 * salinity_psu**2 
               - 4.232e-7 * salinity_psu**3)
    
    esw_0 = esw_00 * a_coeff

    # --- Relaxation Time (Tausw) ---
    tau_sw_0 = (1.1109e-10 
                - 3.824e-12 * temp_celsius 
                + 6.238e-14 * temp_celsius**2 
                - 5.096e-16 * temp_celsius**3)
    
    # Salinity correction factor 'b'
    b_coeff = (1.0 
               + 2.282e-5 * temp_celsius * salinity_psu 
               - 7.638e-4 * salinity_psu 
               - 7.760e-6 * salinity_psu**2 
               + 1.105e-8 * salinity_psu**3)
    
    tau_sw = (tau_sw_0 / (2 * np.pi)) * b_coeff

    # --- Real Part Calculation ---
    omega_tau = 2 * np.pi * freq_hz * tau_sw
    denominator = 1.0 + omega_tau**2
    
    ep_re = esw_inf + ((esw_0 - esw_inf) / denominator)

    # --- Imaginary Part (Conductivity) ---
    # Ionic conductivity term (si)
    si_1 = salinity_psu * (0.18252 
                           - 1.4619e-3 * salinity_psu 
                           + 2.093e-5 * salinity_psu**2 
                           - 1.282e-7 * salinity_psu**3)
    
    delta_t = 25.0 - temp_celsius
    phi = delta_t * (2.033e-2 
                     + 1.266e-4 * delta_t 
                     + 2.464e-6 * delta_t**2 
                     - salinity_psu * (1.849e-5 - 2.551e-7 * delta_t + 2.551e-8 * delta_t**2))
    
    si = si_1 * np.exp(-phi)

    # Debye Imaginary Part + Conductivity Term
    ep_im_numerator = 2 * np.pi * freq_hz * tau_sw * (esw_0 - esw_inf)
    ep_im_debye = ep_im_numerator / denominator
    ep_im_cond = si / (2 * np.pi * EPSILON_0 * freq_hz)
    
    ep_im = ep_im_debye + ep_im_cond
    
    return ep_re, ep_im
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#