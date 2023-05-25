import os
import numpy as np

def get_numpy_asset(name):
    return np.load(os.path.dirname(__file__) + f"/assets/{name}.npy")

def get_lons():
    return get_numpy_asset("longitudes")

def get_lats():
    return get_numpy_asset("latitudes")