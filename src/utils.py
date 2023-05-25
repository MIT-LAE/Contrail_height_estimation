import os
import numpy as np

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
    return get_numpy_asset("ABI2orthographic")