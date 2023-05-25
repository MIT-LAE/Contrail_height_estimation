import os
import numpy as np, datetime as dt
import xarray as xr


def get_numpy_asset(name):
    """
    Loads numpy asset

    Parameters
    ----------
    name : str
        Asset name
    
    Returns
    -------
    array : np.array
        Asset
    """
    return np.load(os.path.dirname(__file__) + f"/assets/{name}.npy")


def get_netcdf_asset(name):
    """
    Loads netcdf asset

    Parameters
    ----------
    name : str
        Asset name
    
    Returns
    -------
    dataset : xr.Dataset
        Asset
    """
    return xr.open_dataset(os.path.dirname(__file__) + f"/assets/{name}.nc")

def get_lons():
    """
    Returns longitudes corresponding to the orthographic projection for
    Meijer et al. (2022) contrail detections.
    """
    return get_numpy_asset("longitudes")

def get_lats():
    """
    Returns latitudes corresponding to the orthographic projection for
    Meijer et al. (2022) contrail detections.
    """
    return get_numpy_asset("latitudes")

def get_ortho_ids():
    """
    Returns the ABI fixed-grid pixel IDs to the orthographic projection for
    Meijer et al. (2022) contrail detections.
    """
    return get_numpy_asset("ABI2orthographic").astype(np.int64)


def floor_time(t, minute_res=10):
    
    minutes = int(np.floor(t.minute/minute_res)*minute_res) % 60
    
    if minutes == 0 and t.minute > (60-minute_res/2):
        return dt.datetime(t.year, t.month, t.day, t.hour+1, minutes)
    else:
        return dt.datetime(t.year, t.month, t.day, t.hour, minutes)