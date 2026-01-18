#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
import numpy as np
import netCDF4 as nc
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that computes the haversine distance between two points on the Earth's surface.
Inputs: lat1 - latitude of the first point
        lon1 - longitude of the first point
        lat2 - latitude of the second point
        lon2 - longitude of the second point
Outputs: d - haversine distance between the two points
"""
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371e3
    t1 = lat1 * (np.pi/180)
    t2 = lat2 * (np.pi/180)
    delta_t = (lat2 - lat1) * (np.pi/180)
    delta_l = (lon2 - lon1) * (np.pi/180)
    sin_dt_2 = np.sin(delta_t/2)
    sin_dl_2 = np.sin(delta_l/2)
    a = (sin_dt_2 * sin_dt_2) + (np.cos(t1) * np.cos(t2) * sin_dl_2 * sin_dl_2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c # in m
    return d
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that computes the indexes of the closest pixels in the SMOS grid for the pixels in the TOPAZ, TSURF and SIC grids. These indexes are used to regrid the data and make them comparable.
Inputs: path_topaz - path to the TOPAZ file
         path_tsurf - path to the TSURF file
         path_smos - path to the SMOS file
         path_sic - path to the SIC file
Outputs: indexs_topaz - indexes of the closest pixels in the SMOS grid for the pixels in the TOPAZ grid
          indexs_tsurf - indexes of the closest pixels in the SMOS grid for the pixels in the TSURF grid
          indexs_sic - indexes of the closest pixels in the SMOS grid for the pixels in the SIC grid
"""
def indexs_regrid(path_topaz, path_tsurf, path_smos, path_sic):
    # Load data - these files may be downloades from public and open sources
    file_topaz=str(path_topaz)
    file_tsurf=str(path_tsurf)
    file_smos=str(path_smos)
    file_sic=str(path_sic)

    # Adequate data
    ds_topaz=nc.Dataset(file_topaz)
    ds_tsurf=nc.Dataset(file_tsurf)
    ds_smos=nc.Dataset(file_smos)
    ds_sic=nc.Dataset(file_sic)
    lat_topaz=np.reshape(np.array(ds_topaz['latitude'][:]),-1)
    lon_topaz=np.reshape(np.array(ds_topaz['longitude'][:]),-1)
    lat_tsurf=np.array(ds_tsurf['lat'][:])
    lon_tsurf=np.array(ds_tsurf['lon'][:])
    lon_tsurf,lat_tsurf=np.meshgrid(lon_tsurf,lat_tsurf)
    lat_tsurf=np.reshape(lat_tsurf,-1)
    lon_tsurf=np.reshape(lon_tsurf,-1)
    lat_smos=np.reshape(np.array(ds_smos['latitude'][:]),-1)
    lon_smos=np.reshape(np.array(ds_smos['longitude'][:]),-1)
    lat_sic=np.reshape(np.array(ds_sic['lat'][:]),-1)
    lon_sic=np.reshape(np.array(ds_sic['lon'][:]),-1)

    # Co-locating the pixels - this may take a while
    indexs_topaz = []
    for k in range(0,len(lon_topaz)):
        print(str(k)+"/"+str(len(lon_topaz)))
        dist = np.abs(haversine_distance(lat_smos,lon_smos,lat_topaz[k],lon_topaz[k]))
        indexs_topaz.append(np.argmin(dist))
    indexs_tsurf=[]
    for k in range(0,len(lon_tsurf)):
        print(str(k)+"/"+str(len(lon_tsurf)))
        dist = np.abs(haversine_distance(lat_smos,lon_smos,lat_tsurf[k],lon_tsurf[k]))
        indexs_tsurf.append(np.argmin(dist))
    indexs_sic=[]
    for k in range(0,len(lon_sic)):
        print(str(k)+"/"+str(len(lon_sic)))
        dist = np.abs(haversine_distance(lat_smos,lon_smos,lat_sic[k],lon_sic[k]))
        indexs_sic.append(np.argmin(dist))

    return indexs_topaz, indexs_tsurf, indexs_sic
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#