#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Burke et al. (1979) Radiative Transfer Model.
"""
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
import numpy as np
from scipy.optimize import root
from permittivity_models import epsilon_snow, epsilon_water # type: ignore
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
def burke_model_ep(theta, dice, tice, sice, snow_presence, epre_ice, epim_ice):
    """
    Computes Brightness Temperature (TB) for horizontal and vertical polarization.
    
    Args:
        theta (float): Incidence angle (degrees).
        dice (float): Sea ice thickness (m).
        tice (float): Sea ice temperature (Celsius).
        sice (float): Sea ice salinity.
        snow_presence (int): 0 for no snow, 1 for snow.
        epre_ice (float): Real part of sea ice permittivity.
        epim_ice (float): Imaginary part of sea ice permittivity.
        
    Returns:
        tuple: (TB_H, TB_V)
    """
    # --- Physical Constants & Parameters ---
    salinity_water = 33      
    temp_water = -1.8        # Celsius
    temp_snow = tice         # Assumption: Snow temp approx. equal to Ice temp
    depth_snow = dice * 0.1  # Assumption: Snow depth is 10% of ice thickness
    rho_snow = 300           # kg/m^3
    freq = 1.4e9             # 1.4 GHz (L-band)
    c_speed = 3e8            # Speed of light (m/s)
    
    theta_rad = np.radians(theta)

    """
    Layer Structure:
    Layer 0: Air
    Layer 1: Snow (if present)
    Layer 2: Sea Ice
    Layer 3: Sea Water
    """

    if snow_presence == 0: 
        # --- 3-Layer Case (Air -> Ice -> Water) ---
        
        # Layer 0: Air
        perm_air_re = 1.0
        perm_air_im = 0.0
        diel_air = complex(perm_air_re, perm_air_im)
        n_air = 1.0
        
        # Layer 1: Ice
        perm_ice_re = epre_ice
        perm_ice_im = epim_ice
        diel_ice = complex(perm_ice_re, perm_ice_im)
        n_ice = np.real(np.sqrt(diel_ice))
        temp_ice_k = tice + 273.15
        
        # Layer 2: Water
        perm_water_re, perm_water_im = epsilon_water(freq, salinity_water, temp_water)
        diel_water = complex(perm_water_re, perm_water_im)
        n_water = np.real(np.sqrt(diel_water))
        temp_water_k = temp_water + 273.15
        
        # --- Propagation Constants ---
        # Air (Layer 0)
        term0 = perm_air_re - np.sin(theta_rad)**2
        beta0 = np.sqrt(0.5 * term0 * (1 + np.sqrt(1 + (perm_air_im**2 / term0**2))))
        alpha0 = perm_air_im / (2 * beta0)
        kappa0 = 2 * np.pi * freq / c_speed * (beta0 + 1j * alpha0)
        
        # Ice (Layer 1)
        theta_ice = np.arcsin(n_air / n_ice * np.sin(theta_rad))
        term1 = perm_ice_re - np.sin(theta_ice)**2
        beta1 = np.sqrt(0.5 * term1 * (1 + np.sqrt(1 + (perm_ice_im**2 / term1**2))))
        alpha1 = perm_ice_im / (2 * beta1)
        gamma1 = 2 * np.pi * freq / c_speed * alpha1 
        kappa1 = 2 * np.pi * freq / c_speed * (beta1 + 1j * alpha1)
        
        # Water (Layer 2)
        theta_water = np.arcsin(n_ice / n_water * np.sin(theta_ice))
        term2 = perm_water_re - np.sin(theta_water)**2
        beta2 = np.sqrt(0.5 * term2 * (1 + np.sqrt(1 + (perm_water_im**2 / term2**2))))
        alpha2 = perm_water_im / (2 * beta2)
        gamma2 = 2 * np.pi * freq / c_speed * alpha2 # Unused in 3-layer reflection but kept for consistency
        kappa2 = 2 * np.pi * freq / c_speed * (beta2 + 1j * alpha2)
        
        # --- Reflection Coefficients ---
        roH_1 = abs((kappa1 - kappa0) / (kappa1 + kappa0))**2.0 
        roH_2 = abs((kappa2 - kappa1) / (kappa2 + kappa1))**2.0
        
        roV_1 = abs((diel_air * kappa1 - diel_ice * kappa0) / (diel_air * kappa1 + diel_ice * kappa0))**2.0
        roV_2 = abs((diel_ice * kappa2 - diel_water * kappa1) / (diel_ice * kappa2 + diel_water * kappa1))**2.0
        
        T_sky = 5
        
        # --- Radiative Transfer Equation (Burke et al. 1979) ---
        # Horizontal Polarization
        A1 = (temp_ice_k * (1 - np.exp(-gamma1 * dice)) * (1 + roH_2 * np.exp(-gamma1 * dice)))
        A2 = temp_water_k
        B1 = (1 - roH_1)
        B2 = (1 - roH_2) * np.exp(-gamma1 * dice)
        
        Tsf = A1 * B1 + A2 * B1 * B2
        TB_RT_H = T_sky * roH_1 + Tsf
        
        # Vertical Polarization
        A1 = (temp_ice_k * (1 - np.exp(-gamma1 * dice)) * (1 + roV_2 * np.exp(-gamma1 * dice)))
        A2 = temp_water_k
        B1 = (1 - roV_1)
        B2 = (1 - roV_2) * np.exp(-gamma1 * dice)
        
        Tsf = A1 * B1 + A2 * B1 * B2
        TB_RT_V = T_sky * roV_1 + Tsf
        
        return TB_RT_H, TB_RT_V
    
    else: 
        # --- 4-Layer Case (Air - Snow - Ice - Water) ---
        
        # Air (Layer 0)
        perm_air_re = 1.0
        perm_air_im = 0.0
        diel_air = complex(perm_air_re, perm_air_im)
        n_air = 1.0
        
        # Snow (Layer 1)
        perm_snow_re, perm_snow_im = epsilon_snow(freq, rho_snow, temp_snow)
        diel_snow = complex(perm_snow_re, perm_snow_im)
        n_snow = np.real(np.sqrt(diel_snow))
        temp_snow_k = temp_snow + 273.15
        
        # Ice (Layer 2)
        perm_ice_re = epre_ice
        perm_ice_im = epim_ice
        diel_ice = complex(perm_ice_re, perm_ice_im)
        n_ice = np.real(np.sqrt(diel_ice))
        temp_ice_k = tice + 273.15
        
        # Water (Layer 3)
        perm_water_re, perm_water_im = epsilon_water(freq, temp_water, salinity_water)   
        diel_water = complex(perm_water_re, perm_water_im)
        n_water = np.real(np.sqrt(diel_water))
        temp_water_k = temp_water + 273.15
        
        # --- Propagation Constants ---
        # Air
        term0 = perm_air_re - np.sin(theta_rad)**2
        beta0 = np.sqrt(0.5 * term0 * (1 + np.sqrt(1 + (perm_air_im**2 / term0**2))))
        alpha0 = perm_air_im / (2 * beta0)
        kappa0 = 2 * np.pi * freq / c_speed * (beta0 + 1j * alpha0)
        
        # Snow
        theta_snow = np.arcsin(n_air / n_snow * np.sin(theta_rad))
        term1 = perm_snow_re - np.sin(theta_snow)**2
        beta1 = np.sqrt(0.5 * term1 * (1 + np.sqrt(1 + (perm_snow_im**2 / term1**2))))
        alpha1 = perm_snow_im / (2 * beta1)
        gamma1 = 2 * np.pi * freq / c_speed * alpha1 
        kappa1 = 2 * np.pi * freq / c_speed * (beta1 + 1j * alpha1)
        
        # Ice
        theta_ice = np.arcsin(n_snow / n_ice * np.sin(theta_snow))
        term2 = perm_ice_re - np.sin(theta_ice)**2
        beta2 = np.sqrt(0.5 * term2 * (1 + np.sqrt(1 + (perm_ice_im**2 / term2**2))))
        alpha2 = perm_ice_im / (2 * beta2)
        gamma2 = 2 * np.pi * freq / c_speed * alpha2
        kappa2 = 2 * np.pi * freq / c_speed * (beta2 + 1j * alpha2)
        
        # Water
        theta_water = np.arcsin(n_ice / n_water * np.sin(theta_ice))
        term3 = perm_water_re - np.sin(theta_water)**2
        beta3 = np.sqrt(0.5 * term3 * (1 + np.sqrt(1 + (perm_water_im**2 / term3**2))))
        alpha3 = perm_water_im / (2 * beta3)
        gamma3 = 2 * np.pi * freq / c_speed * alpha3
        kappa3 = 2 * np.pi * freq / c_speed * (beta3 + 1j * alpha3)
        
        # --- Reflection Coefficients ---
        roH_1 = abs((kappa1 - kappa0) / (kappa1 + kappa0))**2.0 
        roH_2 = abs((kappa2 - kappa1) / (kappa2 + kappa1))**2.0
        roH_3 = abs((kappa3 - kappa2) / (kappa3 + kappa2))**2.0
        
        roV_1 = abs((diel_air * kappa1 - diel_snow * kappa0) / (diel_air * kappa1 + diel_snow * kappa0))**2.0
        roV_2 = abs((diel_snow * kappa2 - diel_ice * kappa1) / (diel_snow * kappa2 + diel_ice * kappa1))**2.0
        roV_3 = abs((diel_ice * kappa3 - diel_water * kappa2) / (diel_ice * kappa3 + diel_water * kappa2))**2.0
        
        T_sky = 5
 
        # --- Radiative Transfer Equation ---
        # Horizontal
        A1 = (temp_snow_k * (1 - np.exp(-gamma1 * depth_snow)) * (1 + roH_2 * np.exp(-gamma1 * depth_snow)))
        A21 = (temp_ice_k * (1 - np.exp(-gamma2 * dice)) * (1 + roH_3 * np.exp(-gamma2 * dice)))
        A22 = temp_water_k
        B1 = (1 - roH_1)
        B2 = (1 - roH_2) * np.exp(-gamma1 * depth_snow)
        B3 = (1 - roH_3) * np.exp(-gamma2 * dice)
        
        Tsf1 = B1 * A1
        Tsf2 = A21 * B1 * B2
        Tsf3 = A22 * B1 * B3 * B2
        Tsf_H = Tsf1 + Tsf2 + Tsf3
        TB_RT_H = T_sky * roH_1 + Tsf_H
        
        # Vertical
        A1 = (temp_snow_k * (1 - np.exp(-gamma1 * depth_snow)) * (1 + roV_2 * np.exp(-gamma1 * depth_snow)))
        A21 = (temp_ice_k * (1 - np.exp(-gamma2 * dice)) * (1 + roV_3 * np.exp(-gamma2 * dice)))
        A22 = temp_water_k
        B1 = (1 - roV_1)
        B2 = (1 - roV_2) * np.exp(-gamma1 * depth_snow)
        B3 = (1 - roV_3) * np.exp(-gamma2 * dice)
        
        Tsf1 = B1 * A1
        Tsf2 = A21 * B1 * B2
        Tsf3 = A22 * B1 * B3 * B2
        Tsf_V = Tsf1 + Tsf2 + Tsf3
        TB_RT_V = T_sky * roV_1 + Tsf_V
        
        return TB_RT_H, TB_RT_V

def burke_model_dretrieval_ep(dice, *conditions):
    """
    Objective function for retrieval.
    """
    theta, I_obs, tice, sice, snow_presence, epre_ice, epim_ice = conditions
    
    # --- Physical Constants & Parameters ---
    salinity_water = 33      
    temp_water = -1.8        
    temp_snow = tice         
    depth_snow = dice * 0.1  
    rho_snow = 300           
    freq = 1.4e9             
    c_speed = 3e8            
    
    theta_rad = np.radians(theta)

    if snow_presence == 0: 
        # --- 3-Layer Case (No Snow) ---
        perm_air_re = 1.0
        perm_air_im = 0.0
        diel_air = complex(perm_air_re, perm_air_im)
        n_air = 1.0
        
        perm_ice_re = epre_ice
        perm_ice_im = epim_ice
        diel_ice = complex(perm_ice_re, perm_ice_im)
        n_ice = np.real(np.sqrt(diel_ice))
        temp_ice_k = tice + 273.15
        
        perm_water_re, perm_water_im = epsilon_water(freq, salinity_water, temp_water)
        diel_water = complex(perm_water_re, perm_water_im)
        n_water = np.real(np.sqrt(diel_water))
        temp_water_k = temp_water + 273.15
        
        # Propagation
        term0 = perm_air_re - np.sin(theta_rad)**2
        beta0 = np.sqrt(0.5 * term0 * (1 + np.sqrt(1 + (perm_air_im**2 / term0**2))))
        alpha0 = perm_air_im / (2 * beta0)
        kappa0 = 2 * np.pi * freq / c_speed * (beta0 + 1j * alpha0)
        
        theta_ice = np.arcsin(n_air / n_ice * np.sin(theta_rad))
        term1 = perm_ice_re - np.sin(theta_ice)**2
        beta1 = np.sqrt(0.5 * term1 * (1 + np.sqrt(1 + (perm_ice_im**2 / term1**2))))
        alpha1 = perm_ice_im / (2 * beta1)
        gamma1 = 2 * np.pi * freq / c_speed * alpha1 
        kappa1 = 2 * np.pi * freq / c_speed * (beta1 + 1j * alpha1)
        
        theta_water = np.arcsin(n_ice / n_water * np.sin(theta_ice))
        term2 = perm_water_re - np.sin(theta_water)**2
        beta2 = np.sqrt(0.5 * term2 * (1 + np.sqrt(1 + (perm_water_im**2 / term2**2))))
        alpha2 = perm_water_im / (2 * beta2)
        gamma2 = 2 * np.pi * freq / c_speed * alpha2
        kappa2 = 2 * np.pi * freq / c_speed * (beta2 + 1j * alpha2)
        
        # Reflection
        roH_1 = abs((kappa1 - kappa0) / (kappa1 + kappa0))**2.0 
        roH_2 = abs((kappa2 - kappa1) / (kappa2 + kappa1))**2.0
        
        roV_1 = abs((diel_air * kappa1 - diel_ice * kappa0) / (diel_air * kappa1 + diel_ice * kappa0))**2.0
        roV_2 = abs((diel_ice * kappa2 - diel_water * kappa1) / (diel_ice * kappa2 + diel_water * kappa1))**2.0
        
        T_sky = 5
        
        # TB Calculation
        A1 = (temp_ice_k * (1 - np.exp(-gamma1 * dice)) * (1 + roH_2 * np.exp(-gamma1 * dice)))
        A2 = temp_water_k
        B1 = (1 - roH_1)
        B2 = (1 - roH_2) * np.exp(-gamma1 * dice)
        Tsf = A1 * B1 + A2 * B1 * B2
        TB_RT_H = T_sky * roH_1 + Tsf
        
        A1 = (temp_ice_k * (1 - np.exp(-gamma1 * dice)) * (1 + roV_2 * np.exp(-gamma1 * dice)))
        A2 = temp_water_k
        B1 = (1 - roV_1)
        B2 = (1 - roV_2) * np.exp(-gamma1 * dice)
        Tsf = A1 * B1 + A2 * B1 * B2
        TB_RT_V = T_sky * roV_1 + Tsf
        
        return ((1/2) * (TB_RT_H + TB_RT_V)) - I_obs
    
    else: 
        # --- 4-Layer Case (With Snow) ---
        perm_air_re = 1.0
        perm_air_im = 0.0
        diel_air = complex(perm_air_re, perm_air_im)
        n_air = 1.0
        
        perm_snow_re, perm_snow_im = epsilon_snow(freq, rho_snow, temp_snow)
        diel_snow = complex(perm_snow_re, perm_snow_im)
        n_snow = np.real(np.sqrt(diel_snow))
        temp_snow_k = temp_snow + 273.15
        
        perm_ice_re = epre_ice
        perm_ice_im = epim_ice
        diel_ice = complex(perm_ice_re, perm_ice_im)
        n_ice = np.real(np.sqrt(diel_ice))
        temp_ice_k = tice + 273.15
        
        perm_water_re, perm_water_im = epsilon_water(freq, temp_water, salinity_water)   
        diel_water = complex(perm_water_re, perm_water_im)
        n_water = np.real(np.sqrt(diel_water))
        temp_water_k = temp_water + 273.15
        
        # Propagation
        term0 = perm_air_re - np.sin(theta_rad)**2
        beta0 = np.sqrt(0.5 * term0 * (1 + np.sqrt(1 + (perm_air_im**2 / term0**2))))
        alpha0 = perm_air_im / (2 * beta0)
        kappa0 = 2 * np.pi * freq / c_speed * (beta0 + 1j * alpha0)
        
        theta_snow = np.arcsin(n_air / n_snow * np.sin(theta_rad))
        term1 = perm_snow_re - np.sin(theta_snow)**2
        beta1 = np.sqrt(0.5 * term1 * (1 + np.sqrt(1 + (perm_snow_im**2 / term1**2))))
        alpha1 = perm_snow_im / (2 * beta1)
        gamma1 = 2 * np.pi * freq / c_speed * alpha1 
        kappa1 = 2 * np.pi * freq / c_speed * (beta1 + 1j * alpha1)
        
        theta_ice = np.arcsin(n_snow / n_ice * np.sin(theta_snow))
        term2 = perm_ice_re - np.sin(theta_ice)**2
        beta2 = np.sqrt(0.5 * term2 * (1 + np.sqrt(1 + (perm_ice_im**2 / term2**2))))
        alpha2 = perm_ice_im / (2 * beta2)
        gamma2 = 2 * np.pi * freq / c_speed * alpha2
        kappa2 = 2 * np.pi * freq / c_speed * (beta2 + 1j * alpha2)
        
        theta_water = np.arcsin(n_ice / n_water * np.sin(theta_ice))
        term3 = perm_water_re - np.sin(theta_water)**2
        beta3 = np.sqrt(0.5 * term3 * (1 + np.sqrt(1 + (perm_water_im**2 / term3**2))))
        alpha3 = perm_water_im / (2 * beta3)
        gamma3 = 2 * np.pi * freq / c_speed * alpha3
        kappa3 = 2 * np.pi * freq / c_speed * (beta3 + 1j * alpha3)
        
        # Reflection
        roH_1 = abs((kappa1 - kappa0) / (kappa1 + kappa0))**2.0 
        roH_2 = abs((kappa2 - kappa1) / (kappa2 + kappa1))**2.0
        roH_3 = abs((kappa3 - kappa2) / (kappa3 + kappa2))**2.0
        
        roV_1 = abs((diel_air * kappa1 - diel_snow * kappa0) / (diel_air * kappa1 + diel_snow * kappa0))**2.0
        roV_2 = abs((diel_snow * kappa2 - diel_ice * kappa1) / (diel_snow * kappa2 + diel_ice * kappa1))**2.0
        roV_3 = abs((diel_ice * kappa3 - diel_water * kappa2) / (diel_ice * kappa3 + diel_water * kappa2))**2.0
        
        T_sky = 5
 
        # TB Calculation
        A1 = (temp_snow_k * (1 - np.exp(-gamma1 * depth_snow)) * (1 + roH_2 * np.exp(-gamma1 * depth_snow)))
        A21 = (temp_ice_k * (1 - np.exp(-gamma2 * dice)) * (1 + roH_3 * np.exp(-gamma2 * dice)))
        A22 = temp_water_k
        B1 = (1 - roH_1)
        B2 = (1 - roH_2) * np.exp(-gamma1 * depth_snow)
        B3 = (1 - roH_3) * np.exp(-gamma2 * dice)
        
        Tsf1 = B1 * A1
        Tsf2 = A21 * B1 * B2
        Tsf3 = A22 * B1 * B3 * B2
        Tsf_H = Tsf1 + Tsf2 + Tsf3
        TB_RT_H = T_sky * roH_1 + Tsf_H
        
        A1 = (temp_snow_k * (1 - np.exp(-gamma1 * depth_snow)) * (1 + roV_2 * np.exp(-gamma1 * depth_snow)))
        A21 = (temp_ice_k * (1 - np.exp(-gamma2 * dice)) * (1 + roV_3 * np.exp(-gamma2 * dice)))
        A22 = temp_water_k
        B1 = (1 - roV_1)
        B2 = (1 - roV_2) * np.exp(-gamma1 * depth_snow)
        B3 = (1 - roV_3) * np.exp(-gamma2 * dice)
        
        Tsf1 = B1 * A1
        Tsf2 = A21 * B1 * B2
        Tsf3 = A22 * B1 * B3 * B2
        Tsf_V = Tsf1 + Tsf2 + Tsf3
        TB_RT_V = T_sky * roV_1 + Tsf_V
        
        return ((1/2) * (TB_RT_H + TB_RT_V)) - I_obs

def d_retrieved_burke_solver_ep(d0, I, theta, tice, sice, snow_presence, epre_ice, epim_ice):
    """
    Inverts the radiative transfer model to retrieve sea ice thickness.
    
    Args:
        d0 (float): Initial thickness guess.
        I (float): Observed intensity.
        ...[other physical params]...
        
    Returns:
        float: Retrieved thickness (m).
    """
    conditions = (theta, I, tice, sice, snow_presence, epre_ice, epim_ice)
    
    # Solve for thickness where (Model_Intensity - Obs_Intensity) == 0
    solve = root(burke_model_dretrieval_ep, d0, args=conditions, method='lm', options={'ftol':10e-2})
    sol = solve.x[0]
    
    # Check for saturation / physical validity
    if sol > 1:
        # Check sensitivity (saturation)
        dice_range = np.arange(1.01, 3.01, 0.01)
        tbh_list = []
        tbv_list = []
        
        # Calculate sensitivity curve
        for d in dice_range:
            h, v = burke_model_ep(theta, d, tice, sice, snow_presence, epre_ice, epim_ice)
            tbh_list.append(h)
            tbv_list.append(v)
            
        tb = (np.array(tbh_list) + np.array(tbv_list)) / 2
        dtb = np.diff(tb)
        
        indices = np.where(np.abs(dtb) < 0.1)[0]
        if indices.size > 0:
            dmax = dice_range[indices[0]]
        else:
            dmax = 1.5
            
        if sol > dmax:
            sol = dmax
            
    if sol < 0:
        sol = 0
        
    return sol
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#