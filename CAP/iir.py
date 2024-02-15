from typing import List
import os
import datetime as dt

import numpy as np
import scipy.constants

from .calipso import CALIPSO

# Central wavelength (in micrometers) of the IIR channels
# Source: Table 2. in Garnier et al. (2018)
# https://amt.copernicus.org/articles/11/2485/2018/amt-11-2485-2018.pdf
IIR_CENTRAL_WAVELENGTH = {1 : 8.621, 2 : 10.635, 3 : 12.058}

# Coefficients to convert IIR Level 1b radiances (in W/m2/micrometer/sr)
# to brightness temperatures (in Kelvin)
# Source: Table 2. in Garnier et al. (2018)
# https://amt.copernicus.org/articles/11/2485/2018/amt-11-2485-2018.pdf
# BT = a0 + (1 + a1) * BT_planck(radiance, central_avelength) 
# units a0: K
# units a1: None
IIR_A0 = {1 : -0.768212, 2 : -0.302290, 3 : -0.466275}
IIR_A1 = {1 : 0.002729, 2 : 0.001314, 3 : 0.002299}


def planck_function(T, wavelength):
    """
    Evaluates the Planck function, which returns the radiance of a blackbody
    at the given temperature and wavelength.

    Parameters
    ----------
    T: float
        Temperature in Kelvin
    wavelength: float
        Wavelength in meters

    Returns
    -------
    B: float
        Radiance in W/m^2/steradian/meter
    """
    # For brevity
    h = scipy.constants.h
    c = scipy.constants.c

    num = 2 * h * c**2 / wavelength**5
    den = np.exp(h * c / (wavelength * scipy.constants.k*T)) - 1

    return num / den

    
def get_brightness_temperature(I, wavelength):
    """
    Obtains the temperature for a blackbody emitting radiance `I` at 
    wavelength `wavelength`. Solves the Planck function for temperature.

    Parameters
    ----------
    I: float
        Radiance in W/m2/meter/sr
    wavelength: float
        Wavelength in meters

    Returns
    -------
    T: float
        Brightness temperature in Kelvin
    """
    # For brevity
    h = scipy.constants.h
    c = scipy.constants.c

    num = h * c
    den = wavelength * scipy.constants.k * np.log(1 + 2 * h * c**2
            / (wavelength**5 * I))
    
    return num / den


class IIR(CALIPSO):
    """
    Class to handle IIR file I/O
    """
    def __init__(self, path):
        """
        Parameters
        ----------
        path : str
            Path to IIR L1 HDF file.
        """
        super().__init__("IIR", path)
        
        self.path = path

    def get_BT_image(self, extent, channel=1, return_coords=True):
        """
        Returns a IIR brightness temperature image

        Parameters
        ----------
        extent : List[float]
            Geodetic extent, [lon_min, lon_max, lat_min, lat_max]
        channel : int, optional
            IIR channel to return, in [1, 2, 3], by default 1
        return_coords : bool, optional
            Whether to return the coordinates associated with the pixels in
            the image, by default True
        
        Returns
        -------
        BTs : np.array
            Brightness temperatures in Kelvin
        """
        if channel not in [1, 2, 3]:
            raise ValueError("Channel must be in [1, 2, 3]")
        
        lons = self.get("Longitude")
        lats = self.get("Latitude")
        mask = (lons >= extent[0]) * (lons <= extent[1]) \
                * (lats >= extent[2]) * (lats <= extent[3])

        # Determine which swath rows are within the geodetic extent
        row_mask = np.where(mask.sum(axis=1) > 0)[0]

        # Subset longitudes and latitudes
        lons = lons[row_mask,:]
        lats = lats[row_mask,:]

        # Get the dataset name corresponding to the IIR channel
        dset_name = "Calibrated_Radiances_"\
                        + ["8.65", "10.6", "12.05"][channel - 1]

        # Get the radiances, which are in units of W/m2/micrometer/sr
        radiances = self.get(dset_name)[row_mask,:]

        # 1e-6 is to convert wavelength from micron to meters
        # 1e6 converts radiances from W/m2/micrometer/sr to W/m2/meter/sr
        BT_planck = get_brightness_temperature(radiances * 1e6,
                        IIR_CENTRAL_WAVELENGTH[channel] * 1e-6)

        # equation 3 in Garnier et al. (2018):
        # https://amt.copernicus.org/articles/11/2485/2018/amt-11-2485-2018.pdf
        BTs = IIR_A0[channel] + (1 + IIR_A1[channel]) * BT_planck
                
        if return_coords:
            return BTs, lons, lats
        else:
            return BTs