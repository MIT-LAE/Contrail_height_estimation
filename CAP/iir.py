from typing import List
import os
import datetime as dt

import numpy as np
import scipy.constants
from pyhdf.SD import SD, SDC


IIR_CENTRAL_WAVELENGTH = {1 : 8.621, 2 : 10.635, 3 : 12.058}
IIR_A0 = {1 : -0.768212, 2 : -0.302290, 3 : -0.466275}
IIR_A1 = {1 : 0.002729, 2 : 0.001314, 3 : 0.002299}


def planck_function(T, lamda):
    """
    Evaluates the Planck function, which returns the radiance of a blackbody
    at the given temperature and wavelength.

    Parameters
    ----------
    T: float
        Temperature in Kelvin
    lamda: float
        Wavelength in micrometers

    Returns
    -------
    B: float
        Radiance in W/m^2/steradian/micrometer
    """
    return 2*((scipy.constants.h*scipy.constants.c**2)/((lamda*1e-6)**5))*\
                1/(np.exp(scipy.constants.h * scipy.constants.c / ((lamda*1e-6)*scipy.constants.k*T)) -1 )


def get_IIR_BT(radiances, channel):
    """
    Obtain the brightness temperature for an Imaging Infrared Radiometer (IIR)
    channel.
    
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
    
    return IIR_A0[channel] + (1 + IIR_A1[channel]) \
        * get_brightness_temperature(radiances, IIR_CENTRAL_WAVELENGTH[channel])
    
    
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


class IIR:
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
        
        self.path = path
        self.file = SD(self.path, SDC.READ)
        self.datasets = [ds for ds in self.file.datasets()]
        #self.time_bounds = self.get_time_bounds(UTC=True)

    def __repr__(self):
        
        return f'IIR object for file: {os.path.basename(self.path)}'
    
    def __str__(self):
        return f'IIR object for file: {os.path.basename(self.path)}'
    
    def available_dataset_names(self):
        """
        Lists available HDF datasets
        """
        return self.datasets
    
    
    def get_time_bounds(self, UTC=True):
        """
        Obtain initial and final time of CALIOP curtain.

        Parameters
        ---------
        UTC : bool (optional)
            To convert time bounds to UTC or not

        Returns
        -------
        time_bounds : List[dt.datetime]
            Initial and final time of CALIOP curtain
        """
        
        if UTC:
            time_key = "Profile_UTC_Time"
        else:
            time_key = "Profile_Time"
        raw_time = self.file.select(time_key)[:]

        time_bounds = [self._convert_time(t) for t in [raw_time[0,0], raw_time[-1, -1]]]
        
        return time_bounds      
        
    def _convert_time(self, t):
        """
        Converts CALIPSO time format to python datetime object
        Based on function found at: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot

        Parameters
        ----------
        t : int
            CALIPSO time format integer
        
        Returns
        -------
        time : dt.datetime
            Datetime object corresponding to CALIPSO time
        """
            
        d = int(t % 100)
        m = int((t-d) % 10000)
        y = int(t-m-d)

        return dt.datetime(2000 + y//10000, m//100, d) + dt.timedelta(t % 1)
    
    def get_time(self, UTC=True):
        """
        Get time dataset of .hdf file and convert to datetime objects.

        Parameters
        ---------
        UTC : bool (optional)
            To convert time bounds to UTC or not

        Returns
        -------
        time : np.array
            Array of datetime values
        """ 

        if UTC:
            time_key = "Profile_UTC_Time"
        else:
            time_key = "Profile_Time"
            
        raw_time = self.file.select(time_key)[:]
        result = [self._convert_time(t) for t in raw_time.flatten()]

        return np.array(result).reshape(raw_time.shape)
    
    def available_dataset_names(self):
        """
        Lists available HDF datasets
        """
        return self.datasets
    
    def get_ground_track(self, n_segments=5):
        lons = self.get("Longitude")[:,0]
        lats = self.get("Latitude")[:,0]
        return GroundTrack(lons, lats, n_segments=n_segments)

    def get(self, dataset, with_units=False):
        """
        Get dataset of .hdf file and fill values if applicable.

        Parameters
        ----------
        dataset : str
            The dataset name
        with_units : bool (optional)
            Returns the dataset units as well

        Returns
        -------
        data : np.array
            Dataset data
        units : str (optional)
            Dataset units
        """ 
        
        if dataset not in self.datasets:
            raise KeyError(f"No dataset found corresponding to {dataset}")

        if not with_units:
            
            # If there is a fillvalue, use this 
            try:
                fill_value = self.file.select(dataset).fillvalue
                return np.ma.masked_equal(self.file.select(dataset)[:], fill_value)

            except:
                return self.file.select(dataset)[:]
     
        else:
            fill_value = self.file.select(dataset).fillvalue
            return np.ma.masked_equal(self.file.select(dataset)[:], fill_value), self.file.select(dataset)["units"]


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