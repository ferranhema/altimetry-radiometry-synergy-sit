#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
RF-sit training script.

This script loads spatial maps of sea ice parameters, performs specific data cleaning
(filtering, sampling, deduplication), and trains the model.

Usage:
    Run as a standalone script to load data and train the model.
    $ python rf-sit_training.py
"""
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_training_data(maps_paths):
    """
    Loads .npy maps from provided directories, concatenates them, and filters
    the data for training.
    """
    # Lists to store flattened data
    dices, tbs, tices, sices, snps, epres, epims = [], [], [], [], [], [], []

    # Iterate over the list of folder paths provided
    for path in maps_paths:
        if not os.path.exists(path):
            print(f"Path not found: {path}")
            continue
            
        maps_dir = np.sort(os.listdir(path))
        print(f"Loading files from: {path}")
        
        for map_file in maps_dir:
            # Skip system files like .DS_Store
            if not map_file.endswith('.npy'):
                continue
                
            print(map_file)
            input_map = np.load(os.path.join(path, map_file))
            
            # Channel mapping:
            # 0: Thickness, 1: TB Intensity, 2: Temperature, 3: Salinity, 
            # 4: Snow, 5: Eps Real, 6: Eps Imag
            dices.append(np.ravel(input_map[0,:,:]))
            tbs.append(np.ravel(input_map[1,:,:]))
            tices.append(np.ravel(input_map[2,:,:]))
            sices.append(np.ravel(input_map[3,:,:]))
            snps.append(np.ravel(input_map[4,:,:]))
            epres.append(np.ravel(input_map[5,:,:]))
            epims.append(np.ravel(input_map[6,:,:]))

    # Concatenate all lists into numpy arrays
    dices = np.concatenate(dices)
    tbs = np.concatenate(tbs)
    tices = np.concatenate(tices)
    sices = np.concatenate(sices)
    snps = np.concatenate(snps)
    epres = np.concatenate(epres)
    epims = np.concatenate(epims)

    # Filter invalid data (-999)
    valid_mask = tbs != -999
    
    dice_train = dices[valid_mask]
    tice_train = tices[valid_mask]
    sice_train = sices[valid_mask]
    snp_train = snps[valid_mask]
    epre_train = epres[valid_mask]
    epim_train = epims[valid_mask]
    i_train = tbs[valid_mask]

    # Stack data for DataFrame
    data = np.column_stack((i_train, tice_train, sice_train, snp_train, epre_train, epim_train))

    data_train_burke = pd.DataFrame(
        data=data, 
        columns=['intensity', 'temperature', 'salinity', 'snow presence', 'epsilon real', 'epsilon imaginary']
    )
    data_train_burke['thickness'] = dice_train
    
    # Specific Sampling and Deduplication Logic
    data_train_burke = data_train_burke.sample(frac=0.005, random_state=42).reset_index(drop=True)

    data_train_burke = data_train_burke.drop_duplicates(subset=['intensity'])
    data_train_burke = data_train_burke.drop_duplicates(subset=['temperature'])
    data_train_burke = data_train_burke.drop_duplicates(subset=['salinity'])

    # Slice specific rows
    data_train_burke = data_train_burke.iloc[100000:200000, :]

    return data_train_burke

def rf_sit_train(data_train_burke):
    """
    Trains the RF-sit model.
    """
    data_train, data_test = train_test_split(data_train_burke, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=50, criterion='squared_error', random_state=42)
    
    # Features (cols 0-5), Target (col 6: thickness)
    rf.fit(data_train.iloc[:,:6], data_train.iloc[:,6])

    return rf
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    base_path = './input_maps/input_maps_ep'
    maps_path_1 = os.path.join(base_path, '1920')
    maps_path_2 = os.path.join(base_path, '2021')
    
    # Pass both paths as a list
    data_train = get_training_data([maps_path_1, maps_path_2])
    
    if not data_train.empty:
        rf_model = rf_sit_train(data_train)
        print("Model trained successfully.")
    else:
        print("Dataframe is empty after processing.")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#