#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
import numpy as np
import netCDF4 as nc
import pandas as pd
import random
import time
import os
from burke_model import d_retrieved_burke_solver_ep # type: ignore
from sice_empirical import sice_empirical # type: ignore
from ellipsoids_perm import e_eff_mix # type: ignore
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function to generate the input maps for the models from the data of the different sources. The function reads the data from the different sources, 
filters the data, and generates the input maps for the models. The function returns the input maps. The function receives the period, the path to the 
data of the TOPAZ model, the path to the data of the CMEMS model, the path to the data of the SMOS model, and the path to the data of the SIC model.
Inputs:
    - period: period of the data
    - path_topaz: path to the data of the TOPAZ model
    - path_tsurf: path to the data of the CMEMS model
    - path_smos: path to the data of the SMOS model
    - path_sic: path to the data of the SIC model
Outputs:
    - input_map: input maps for the models  
"""
def map_generation(period, path_topaz, path_tsurf, path_smos, path_sic):
    start_time = time.time()
    files_topaz=np.sort(os.listdir(str(path_topaz)+"/"+str(period)+"/"))    
    file_tsurf=str(path_tsurf)+"/cmems_obs_si_arc_phy_my_L4-DMIOI_P1D-m_"+str(period)+".nc"
    files_smos=np.sort(os.listdir(str(path_smos)+"/"+str(period)+"/"))
    files_sic=np.sort(os.listdir(str(path_sic)+"/"+str(period)+"/"))
    c = 0
    for k in range(0,len(files_smos)):
        print(str(k)+"/"+str(len(files_smos)))
        ds_tsurf=nc.Dataset(file_tsurf)
        ds_topaz=nc.Dataset(str(path_topaz)+"/"+str(period)+"/"+str(files_topaz[k]))
        ds_smos=nc.Dataset(str(path_smos)+"/"+str(period)+"/"+str(files_smos[k]))
        ds_sic=nc.Dataset(str(path_sic)+"/"+str(period)+"/"+str(files_sic[k]))

        lat_topaz=np.array(ds_topaz['latitude'][:])
        lon_topaz=np.array(ds_topaz['longitude'][:])
        lat_tsurf=np.array(ds_tsurf['latitude'][:])
        lon_tsurf=np.array(ds_tsurf['longitude'][:])
        lat_smos=np.array(ds_smos['latitude'][:])
        lon_smos=np.array(ds_smos['longitude'][:])
        lat_sic=np.array(ds_sic['lat'][:])
        lon_sic=np.array(ds_sic['lon'][:])

        sss_topaz=np.array(ds_topaz['so'][0,0,:,:])
        sit_topaz=np.array(ds_topaz['sithick'][0,:,:])
        snow_topaz=np.array(ds_topaz['sisnthick'][0,:,:])
        sic=np.array(ds_sic['ice_conc'][0,:,:])
        tsurf=np.array(ds_tsurf['analysed_st'][k,:,:])
        i_smos=np.array(ds_smos['TB'][0,:,:])

        df_tsurf = pd.read_csv('.../indexs/indexs_tsurf.csv',index_col=0)
        tsurf = np.reshape(tsurf, -1)
        df_tsurf['tsurf'] = tsurf
        i_smos0 = np.reshape(i_smos,-1)
        df_smos = pd.DataFrame()
        df_smos['i_smos'] = i_smos0
        df_smos['myid_tsurf'] = df_smos.index
        df_smos = df_smos.drop(columns=['i_smos'])
        df = pd.merge_asof(df_smos, df_tsurf.sort_values('myid_tsurf'), on=['myid_tsurf'])
        tsurf_rg = np.reshape(df['tsurf'].to_numpy(),(896,608))

        df_sic = pd.read_csv('.../indexs/indexs_sic.csv',index_col=0)
        sic = np.reshape(sic, -1)
        df_sic['sic'] = sic
        i_smos0 = np.reshape(i_smos,-1)
        df_smos = pd.DataFrame()
        df_smos['i_smos'] = i_smos0
        df_smos['myid_sic'] = df_smos.index
        df_smos = df_smos.drop(columns=['i_smos'])
        df = pd.merge_asof(df_smos, df_sic.sort_values('myid_sic'), on=['myid_sic'])
        sic_rg = np.reshape(df['sic'].to_numpy(),(896,608))

        df_topaz = pd.read_csv('.../indexs/indexs_topaz.csv',index_col=0)
        sss_topaz = np.reshape(sss_topaz, -1)
        df_topaz['sss_topaz'] = sss_topaz
        i_smos0 = np.reshape(i_smos,-1)
        df_smos = pd.DataFrame()
        df_smos['i_smos'] = i_smos0
        df_smos['myid_topaz'] = df_smos.index
        df_smos = df_smos.drop(columns=['i_smos'])
        df = pd.merge_asof(df_smos, df_topaz.sort_values('myid_topaz'), on=['myid_topaz'])
        sss_topaz_rg = np.reshape(df['sss_topaz'].to_numpy(),(896,608))
        
        df_topaz = pd.read_csv('.../indexs/indexs_topaz.csv',index_col=0)
        sit_topaz = np.reshape(sit_topaz, -1)
        df_topaz['sit_topaz'] = sit_topaz
        i_smos0 = np.reshape(i_smos,-1)
        df_smos = pd.DataFrame()
        df_smos['i_smos'] = i_smos0
        df_smos['myid_topaz'] = df_smos.index
        df_smos = df_smos.drop(columns=['i_smos'])
        df = pd.merge_asof(df_smos, df_topaz.sort_values('myid_topaz'), on=['myid_topaz'])
        sit_topaz_rg = np.reshape(df['sit_topaz'].to_numpy(),(896,608))
        
        df_topaz = pd.read_csv('.../indexs/indexs_topaz.csv',index_col=0)
        snow_topaz = np.reshape(snow_topaz, -1)
        df_topaz['snow_topaz'] = snow_topaz
        i_smos0 = np.reshape(i_smos,-1)
        df_smos = pd.DataFrame()
        df_smos['i_smos'] = i_smos0
        df_smos['myid_topaz'] = df_smos.index
        df_smos = df_smos.drop(columns=['i_smos'])
        df = pd.merge_asof(df_smos, df_topaz.sort_values('myid_topaz'), on=['myid_topaz'])
        snow_topaz_rg = np.reshape(df['snow_topaz'].to_numpy(),(896,608))

        mask_ssstopaz = np.logical_or(sit_topaz_rg < 0, np.isnan(sit_topaz_rg))
        sss_topaz_rg[mask_ssstopaz] = 33
        mask_sittopaz = np.logical_or(sit_topaz_rg < 0, np.isnan(sit_topaz_rg))
        sit_topaz_rg[mask_sittopaz] = 0.1
        mask_snowtopaz = np.logical_or(snow_topaz_rg < 0, np.isnan(snow_topaz_rg))
        snow_topaz_rg[mask_snowtopaz] = 0
        mask_tsurf = np.logical_or(tsurf_rg < 200, np.isnan(tsurf_rg))
        tsurf_rg[mask_tsurf] = 250
        condition_mask = (
            (i_smos >= 0) &
            (sic_rg > 0) &
            ((lat_smos > 58) | ((lon_smos < -100) | (lon_smos > -70)))
        )
        sit_test = sit_topaz_rg[condition_mask]
        sice_test = sice_empirical(sss_topaz_rg[condition_mask], sit_topaz_rg[condition_mask])
        tice_test = ((tsurf_rg[condition_mask] - 273.15) - 1.8) / 2
        i_test = i_smos[condition_mask]
        lat_test = lat_smos[condition_mask]
        lon_test = lon_smos[condition_mask]
        sss_test = sss_topaz_rg[condition_mask]
        snow_test = snow_topaz_rg[condition_mask]
        snp_test = np.where(snow_test > 0, 1, 0)
        sic_test = sic_rg[condition_mask]

        preds_invb = []
        epres = []
        epims = []
        lenght_ratios_1 = np.arange(0.001, 10, 0.001)
        lenght_ratios_2 = np.arange(10, 1000, 0.5)
        lenght_ratios = np.concatenate((lenght_ratios_1, lenght_ratios_2))
        for i in range(0,len(i_test)):
            epre, epim = e_eff_mix(random.choice(lenght_ratios), 0, 2, sice_test[i], tice_test[i])
            epres.append(epre)
            epims.append(epim)
            preds_invb.append(d_retrieved_burke_solver_ep(0, i_test[i], 40, tice_test[i], sice_test[i], snp_test[i], epre, epim))
        input_map=np.zeros((5,896,608))

        coord_dict0 = {}
        coord_dict1 = {}
        coord_dict2 = {}
        coord_dict3 = {}
        coord_dict4 = {}
        coord_dict5 = {}
        coord_dict6 = {}
        for l in range(len(lon_test)):
            coord_dict0[(lon_test[l], lat_test[l])] = preds_invb[l]
            coord_dict1[(lon_test[l], lat_test[l])] = i_test[l]
            coord_dict2[(lon_test[l], lat_test[l])] = tice_test[l]
            coord_dict3[(lon_test[l], lat_test[l])] = sice_test[l]
            coord_dict4[(lon_test[l], lat_test[l])] = snp_test[l]
            coord_dict5[(lon_test[l], lat_test[l])] = epres[l]
            coord_dict6[(lon_test[l], lat_test[l])] = epims[l]
        for i in range(0, 896):
            for j in range(0, 608):
                coord = (lon_smos[i][j], lat_smos[i][j])
                if coord in coord_dict0:
                    input_map[0][i][j] = coord_dict0[coord]
                if coord in coord_dict1:
                    input_map[1][i][j] = coord_dict1[coord]        
                if coord in coord_dict2:
                    input_map[2][i][j] = coord_dict2[coord]        
                if coord in coord_dict3:
                    input_map[3][i][j] = coord_dict3[coord]        
                if coord in coord_dict4:
                    input_map[4][i][j] = coord_dict4[coord]
                if coord in coord_dict5:
                    input_map[5][i][j] = coord_dict5[coord]
                if coord in coord_dict6:
                    input_map[6][i][j] = coord_dict6[coord]
        print("--- %s seconds ---" % (time.time() - start_time))
        c += 1
    return input_map