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
    


def get_image_tile_indices(rows, cols, region_size, image_shape):
    """
    For given input locations (representing the centers of the tiels) within an image
    and a 'tile size', will return the indices within the image to create these tiles.
    
    Parameters
    ----------
    rows : np.array
        Row locations of tile centers
    cols : np.array
        Column locations of tile centers
    region_size : int
        Size of tiles
    image_shape : np.array
        The shape of the image from which the tiles will be taken
        
    Returns
    -------
    reg_rows : np.array
        Array of dimensions region_size**2 by len(rows)  
    reg_cols : np.array
        Array of dimensions region_size**2 by len(cols)
    """
    
    if region_size % 2 == 0:
        raise ValueError("Region size needs to be odd")
    
    # Stack copies on top of each other to represent the regional indices
    reg_cols = np.tile(cols, (region_size**2, 1))
    reg_rows = np.tile(rows, (region_size**2, 1))

    offsets = np.arange(-(region_size-1)//2,  1+(region_size-1)//2)

    reg_cols += np.tile(offsets, (region_size))[:,np.newaxis]
    reg_rows += np.repeat(offsets, (region_size))[:, np.newaxis]

    # Find illegal indices
    top_margins = reg_rows.min(axis=0)
    reg_rows[:,top_margins < 0] -= top_margins[top_margins < 0]
    bot_margins = image_shape[0] - reg_rows.max(axis=0) - 1
    reg_rows[:,bot_margins < 0] += bot_margins[bot_margins < 0]
    
    left_margins = reg_cols.min(axis=0)
    reg_cols[:,left_margins < 0] -= left_margins[left_margins < 0]
    right_margins = image_shape[1] - reg_cols.max(axis=0) - 1
    reg_cols[:,right_margins < 0] += right_margins[right_margins < 0]
    
    return reg_rows, reg_cols
    