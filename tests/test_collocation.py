import numpy as np
from src.collocation import *
import pytest
import xarray as xr, datetime as dt


def test_geometricaltitude2pressure():

    # Create fake weather dataset
    pressures = np.array([450., 400., 350., 300., 250., 200.])
    lons = np.array([-130., -125.0])
    lats = np.array([30., 35.])
    times = np.array([dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)])

    # Geopotential altitude
    z = 9.81* 100000 * np.exp(-pressures[np.newaxis,np.newaxis,:,np.newaxis]/100.) \
        * np.ones((len(lons), len(lats), len(pressures), len(times)))
    
    ds = xr.Dataset({"z": (["longitude", "latitude", "isobaricInhPa","time"], z)},
    coords={"longitude": lons, "latitude": lats, "isobaricInhPa": pressures, "time": times})

    p = geometricaltitude2pressure(-127.5, 32.5, dt.datetime(2020, 1, 1, 12), 8e3, ds)
    analytical = -100*np.log(8/100.)
    assert pytest.approx(p,rel=1e-1) == analytical


def test_parallax_correction_vicente():

    




