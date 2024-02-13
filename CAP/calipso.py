"""
Base class for CALIPSO instrument dataset processing
"""
import os
import datetime as dt

import numpy as np
from pyhdf.SD import SD, SDC

from .geometry import GroundTrack

class CALIPSO:

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.file = SD(self.path, SDC.READ)
        self.datasets = [ds for ds in self.file.datasets()]

    def __repr__(self):
        return f'{self.name} object for file: {os.path.basename(self.path)}'
    
    def __str__(self):
        return f'{self.name} object for file: {os.path.basename(self.path)}'


    def available_dataset_names(self):
        """
        Lists available HDF datasets
        """
        return self.datasets
    
    def get_time_bounds(self, UTC=True):
        """
        Obtain initial and final time of data.

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

        time_bounds = [self._convert_time(t) for t in [raw_time[0,0],
                        raw_time[-1, -1]]]
        
        return time_bounds      
        
    def _convert_time(self, t):
        """
        Converts CALIPSO time format to python datetime object
        Based on function found at:
        https://github.com/peterkuma/ccplot/blob/master/bin/ccplot

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
    
    def get_ground_track(self, n_segments=5):
        """
        Get CALIPSO ground track corresponding to instrument data.
        `n_segments` denotes the number of great circle paths used to 
        approximate the ground track.
        """
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

