import numpy as np
import datetime as dt

from pyhdf.SD import SD, SDC
from pyhdf import HDF, VS


from vertical_feature_mask import interpret_vfm
from contrails.numerics.algorithms import *

PROJECTION_PATH = '/net/d13/data/vmeijer/reprojections/'

CALIOP_ALTITUDES = np.hstack((np.arange(40.0, 30.1, -0.300),
                       np.arange(30.1, 20.1, -0.180),
                       np.arange(20.1, 8.5, -0.060),
                       np.arange(8.5, -0.5, -0.030)))


class CollocationError(Exception):
    """Exception raised for errors with collocation """
    
    pass


class CALIOP:
    """
    Class to handle CALIOP file I/O
    """
    def __init__(self, path):
        """
        Parameters
        ----------
        path : str
            Path to CALIOP L1/L2 HDF file.
        """
        
        self.path = path
        self.file = SD(self.path, SDC.READ)
        self.datasets = [ds for ds in self.file.datasets()]
        self.time_bounds = self.get_time_bounds(UTC=True)
        self.read_meta_data()
        
    def read_meta_data(self):
        """
        Reads HDF metadata.
        """

        h4 =  HDF.HDF(self.path)
        vs = h4.vstart()
        vs_meta = vs.attach('metadata')
        field_names = vs_meta.inquire()[2]

        for i, name in enumerate(field_names):
            setattr(self, name, vs_meta[0][i])
        h4.close()

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

    def is_ascending(self):
        """
        Checks whether this is an ascending orbit (i.e. daytime overpass)
        or a descending orbit (i.e. nighttime overpass).
        """
        lats = self.get("Latitude")
        
        if lats[1,0] - lats[0,0] > 0:
            ascending = True
        else:
            ascending = False 

        return ascending
    
    def filter_rows(self, lat_min, lat_max, cirrus=True):
        """
        Returns mask that can be used for indexing any dataset in order to
        subset it to the given latitude range.

        Parameters
        ----------
        lat_min : float
            Minimum latitude, degrees
        lat_max : float
            Maximum latitude, degrees
        cirrus : bool (optional)
            Whether or not to filter for cirrus data only (L2 feature)
        """

        lats = self.get("Latitude")
        fcfs = self.get("Feature_Classification_Flags")
        extinction_qcs = self.get("ExtinctionQC_532")
        top_alts = self.get("Layer_Top_Altitude")

        # Find coordinates of endpoints of (sub-)trajectory
        # Column 0 contains locations corresponding to start of a 5km segment
        # Column 2 contains locations corresponding to end of a 5km segment
        if self.is_ascending():
            rows = np.where((lats[:,0] >= lat_min)*(lats[:,2] <= lat_max))[0]
        else:
            rows = np.where((lats[:,2] >= lat_min)*(lats[:,0] <= lat_max))[0]

        # Filter rows based on feature classification flags and extinction quality control
        rows_to_keep = []
        for r in rows[~top_alts.mask[rows, 0]]:
            fcf = fcfs[r, 0]
            extinctionqc = extinction_qcs[r,0]

            if not filter_cirrus_feature(fcf) or extinctionqc >= 16:
                continue 

            rows_to_keep.append(r)

        if not cirrus:
            rows_to_keep = rows[~np.isin(rows, rows_to_keep)]

        return rows_to_keep


def filter_cirrus_feature(fcf):
    """
    Given a feature classification flag from the CALIOP L2 layer product,
    determines whether the feature is cirrus cloud.

    Parameters
    ----------
    fcf : int
        Feature classification flag

    Returns
    -------
    is_cirrus : bool
        Whether the feature is cirrus
    """
    
    interpreted = interpret_vfm(fcf)

    conditions = [interpreted[0] == "cloud", interpreted[1] == "high",
                  interpreted[2] in ["randomly oriented ice", "horizontally oriented ice"],
                  interpreted[3] == "high"]

    return all(conditions)

    
def interpolate_caliop_profile(data, lidar_alts=CALIOP_ALTITUDES, ve2=40.0e3, ve1=0.0, vres=30.0):
    """
    Interpolates a CALIOP L1 profile to a regular grid.

    Parameters
    ----------
    data: np.array
        Data to interpolate
    lidar_alts: np.array (optional)
        The altitude (MSL) corresponding to the rows of 'data', in km
    ve2: float (optional)
        Vertical coordinate of top of interpolated grid in meters
    ve1: float (optional)
        Vertical coordinate of bottom of interpolated grid in meters
    vres: float (optional)
        The vertical resolution to interpolate to

    Returns
    -------
    interpolated: np.array
        Interpolated data
    """

    e1 = 0
    e2 = data.shape[0]
    
    X = np.arange(e1, e2, dtype=np.float32)
    Y = np.meshgrid(1000*lidar_alts, X)[0].astype(np.float32)
    data = np.array(np.ma.masked_equal(data,  -9999).astype(np.float32))
    interpolated = interp2d_12(data.astype(np.float32), X, Y, float(e1), float(e2), int(e2-e1), ve2, ve1,
                       int((ve2-ve1)/vres))
    
    return interpolated


def subset_caliop_profile(caliop, dataset_name, extent,
                             return_coords=False):
    """
    Subset a particular CALIOP profile to a geodetic extent.

    Parameters
    ----------
    caliop: CALIOP
        CALIOP object containing data to subset
    dataset_name: str
        The name of the variable to subset
    extent: list
        List of minimum longitude, maximum longitude, minimum latitude,
        maximum latitude in degrees
    return_coords: bool (optional)
        Option to return longitudes, latitudes, times of subsetted profile
    
    Returns
    -------
    subset_data: np.array
        Subsetted data
    """
    
    # Extract longitudes, latitudes and UTC times
    lons = caliop.get("Longitude")
    lats = caliop.get("Latitude")
    times = caliop.get_time()
    
    # Extract data
    data = caliop.get(dataset_name)
    
    # Now apply subsetting
    subset_mask = (lons >= extent[0])*(lons <= extent[1])\
                    *(lats >= extent[2])*(lats <= extent[3])
    subset_data = data[subset_mask[:,0], :]
    
    if return_coords:
        return subset_data, lons[subset_mask], lats[subset_mask], times[subset_mask]
        
    else:
        return subset_data
    
