#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
RF-eps algorithm training using SMRT simulations.

This script generates a synthetic dataset of ku-band radar altimeter backscatter using the 
Snow Microwave Radiative Transfer (SMRT) model and trains a Random Forest 
Regressor to retrieve sea ice L-band permittivity.

Usage:
    Run as a standalone script to generate data and train the model.
    $ python train_permittivity_model.py
"""
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
from tqdm import tqdm  # Progress bar for long simulations

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ellipsoids_perm import e_eff_mix  # type: ignore
from smrt import make_snowpack, make_ice_column, make_model, make_interface, PSU  # type: ignore 
from smrt.inputs import altimeter_list  # type: ignore
from smrt.emmodel.iba import derived_IBA  # type: ignore
from smrt.permittivity.generic_mixing_formula import polder_van_santen_mod  # type: ignore
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
def rf_eps_training_dataset_generation():
    """
    Generates a training dataset by running SMRT simulations over a range of parameters.

    The function simulates the difference in backscatter between a reference
    sea ice column and one with a rough interface.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'backscatter': The simulated backscatter difference (dB).
            - 'tice': Sea Ice temperature (K).
            - 'sice': Sea Ice salinity.
            - 'epre': Real part of permittivity.
            - 'epim': Imaginary part of permittivity.
    """
    # 1. Define Sensor and Model
    sensor = altimeter_list.cryosat2_sin()
    altimodel = make_model(derived_IBA(polder_van_santen_mod), "nadir_lrm_altimetry")
    rough_interface = make_interface("geometrical_optics_backscatter", mean_square_slope=0.03)
    
    # 2. Define Snowpack (Static)
    snow = make_snowpack(
        thickness=[0.1], 
        microstructure_model='exponential', 
        length_ratio=1,
        density=[300], 
        corr_length=0.5e-4, 
        temperature=268, 
        salinity=[0]
    )

    # 3. Define Parameter Ranges
    tices = np.arange(255, 270, 0.5)  # Temperature in Kelvin
    sices = np.arange(3, 10, 0.5)     # Salinity in PSU
    
    # Brine inclusion length ratios
    length_ratios_1 = np.arange(0.001, 0.01, 0.005)
    length_ratios_2 = np.arange(0.01, 0.1, 0.05)
    length_ratios_3 = np.arange(0.1, 1, 0.5)
    length_ratios_4 = np.arange(1, 10, 0.5)
    length_ratios_5 = np.arange(10, 1000, 5)
    length_ratios = np.concatenate((
        length_ratios_1, length_ratios_2, length_ratios_3, 
        length_ratios_4, length_ratios_5
    ))

    # 4. Initialize Storage
    maxs_ell = []
    epre_ell = []
    epim_ell = []
    tice_ell = []
    sice_ell = []

    # Calculate total iterations for progress bar
    total_iters = len(tices) * len(sices) * len(length_ratios)
    print(f"Starting SMRT simulation with {total_iters} combinations...")

    # 5. Simulation Loop
    # Using tqdm to show progress bar in console
    with tqdm(total=total_iters) as pbar:
        for tice in tices:
            for sice in sices:
                for lr in length_ratios:
                    # Calculate complex permittivity (temperature converted to Celsius)
                    epre, epim = e_eff_mix(lr, 0, 2, sice, tice - 273.15)
                    
                    # Create Sea Ice Column
                    ice_ell = make_ice_column(
                        ice_type='firstyear',
                        thickness=[1], 
                        temperature=tice, 
                        microstructure_model='independent_sphere',
                        radius=1e-3,
                        length_ratio=[lr],
                        density=900,
                        salinity=sice * PSU,
                        add_water_substrate=True
                    )
                    
                    # Construct Mediums
                    medium_ell = snow + ice_ell
                    
                    # Create calibration medium (copy of original)
                    medium_cal_ell = medium_ell.copy()
                    
                    # Apply rough interface to the main medium
                    medium_ell.interfaces[-1] = rough_interface
                    
                    # Run Model
                    result_cal_ell = altimodel.run(sensor, medium_cal_ell)
                    result_ell = altimodel.run(sensor, medium_ell)
                    
                    # Compute Backscatter Difference
                    sigma_diff = np.nanmax(result_cal_ell.sigma_dB()) - np.nanmax(result_ell.sigma_dB())
                    
                    # Store Results
                    maxs_ell.append(sigma_diff)
                    epre_ell.append(epre)
                    epim_ell.append(epim)
                    tice_ell.append(tice)
                    sice_ell.append(sice)
                    
                    pbar.update(1)

    # 6. Create DataFrame
    train_ep_sig = pd.DataFrame({
        'backscatter': maxs_ell,
        'tice': tice_ell,
        'sice': sice_ell,
        'epre': epre_ell,
        'epim': epim_ell
    })
    
    return train_ep_sig

def rf_eps_train(training_dataset):
    """
    Trains a the RF-eps algorithm on the generated dataset.
    
    Note: This function scales the 'backscatter' feature internally before training.
    
    Args:
        training_dataset (pd.DataFrame): The dataframe output from `rf_eps_training_dataset_generation`.
        
    Returns:
        sklearn.ensemble.RandomForestRegressor: The trained RF-eps model.
    """
    train_ep_sig = training_dataset.copy()
    
    # Scale the backscatter
    scaler = StandardScaler()
    scaler.fit(train_ep_sig[['backscatter']])
    train_ep_sig['backscatter'] = scaler.transform(train_ep_sig[['backscatter']])
    
    # Initialize and Train RF-eps
    rf = RandomForestRegressor(
        n_estimators=50, 
        criterion='squared_error', 
        random_state=42
    )
    
    # Features: backscatter (scaled), tice, sice
    # Targets: epre, epim
    rf.fit(train_ep_sig.iloc[:, :3], train_ep_sig.iloc[:, 3:])
    
    return rf
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    # Generate the dataset
    print("Generating training data...")
    train_ep_sig = rf_eps_training_dataset_generation()
    
    # Optional: Save the dataset
    # train_ep_sig.to_csv('training_data.csv', index=False)
    
    # Train the model
    print("Training Random Forest model...")
    rf_model = rf_eps_train(train_ep_sig)
    
    print("Process complete. Model trained.")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#