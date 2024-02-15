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


def get_IIR_BT(radiances, channel):
    """
    Obtain the brightness temperature for an Imaging Infrared Radiometer (IIR)
    channel.
    
    Based on equation 3 in Garnier et al. (2018):
    https://amt.copernicus.org/articles/11/2485/2018/amt-11-2485-2018.pdf
    
    Parameters
    ----------
    radiances : np.array
        Radiance in W/m2/micrometer/sr
    channel : int
        Channel, in [1, 2, 3]
    
    Returns
    -------
    BTs : np.array
        Brightness temperatures in Kelvin
    """
    BT_planck = get_brightness_temperature(radiances,
        IIR_CENTRAL_WAVELENGTH[channel])
    return IIR_A0[channel] + (1 + IIR_A1[channel]) * BT_planck
    
    
def get_brightness_temperature(I, lamda):
    """
    Obtains the temperature for a blackbody emitting radiance `I` at 
    wavelength `lamda`.

    Parameters
    ----------
    I: float
        Radiance in W/m2/micrometer/sr
    lamda: float
        Wavelength in micrometers

    Returns
    -------
    T: float
        Brightness temperature in Kelvin
    """
    lamda = lamda * 1e-6
    return scipy.constants.h * scipy.constants.c / (lamda * scipy.constants.k * np.log(1 + 2 * scipy.constants.h * scipy.constants.c**2 / (lamda**5 * I)))


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

   
def get_IIR_image(caliop, extent, channel=1, return_coords=True):
    
    l1_path = caliop.path

    if "2021" in l1_path or "2022" in l1_path:
        v = "V2-01"
    else:
        v = "V2-00"
    
    iir_path = "/net/d15/data/vmeijer/IIR_L1/" + os.path.basename(
    l1_path.replace("CALIOP_L1", "IIR_L1").replace("V4-51", v).replace("V4-10", v).replace("V4-11", v).replace("LID", "IIR").replace("_Subset.hdf",".hdf").replace("ValStage1-V3-41", "Standard-V2-01"))
      
    try:
        iir = IIR(iir_path)
    except Exception as e:
        print(f"Something wrong with file at {iir_path}")
        raise e

    iir_lons = iir.get("Longitude")
    iir_lats = iir.get("Latitude")
    mask = (iir_lons >= extent[0])*(iir_lons <= extent[1]) * (iir_lats >= extent[2]) * (iir_lats <= extent[3])

    row_mask = np.arange(mask.shape[0])
    row_mask = row_mask[~np.isin(row_mask, np.where(mask.sum(axis=1) == 0)[0])]


    iir_lons = iir_lons[row_mask,:]
    iir_lats = iir_lats[row_mask,:]

    rad_name = ["8.65", "10.6", "12.05"][channel-1]

    rads = iir.get("Calibrated_Radiances_"+rad_name)[row_mask,:]
    
    # The 1000 is the scale factor
    BTs = get_IIR_BT(1000*rads.data.astype(np.float64), channel)
    if return_coords:
        return BTs, iir_lons, iir_lats
    else:
        return BTs