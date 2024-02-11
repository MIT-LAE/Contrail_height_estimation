import numpy as np
import datetime as dt
import os

from pyhdf.SD import SD, SDC
from pyhdf import HDF, VS
import skimage.measure

from .calipso import CALIPSO
from .vertical_feature_mask import interpret_vfm
from .visualization import plot_caliop_profile_direct, loadcolormap
from .interpolation import interp2d_12

PROJECTION_PATH = '/net/d13/data/vmeijer/reprojections/'

CALIOP_ALTITUDES = np.hstack((np.arange(40.0, 30.1, -0.300),
                       np.arange(30.1, 20.1, -0.180),
                       np.arange(20.1, 8.5, -0.060),
                       np.arange(8.5, -0.5, -0.030)))



# See https://www.eoportal.org/satellite-missions/calipso#
# Refers to L1b product resolution
CALIOP_HORIZONTAL_RESOLUTION = 333 # m
CALIOP_VERTICAL_RESOLUTION = 30 # m

# See Iwabuchi et al. (2012), section 3.3:
# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JD017020
BACKSCATTER_THRESHOLD = 0.003 # km^-1 sr^-1
WIDTH_THRESHOLD = 1000 # meter
THICKNESS_THRESHOLD = 60 # meter


class CollocationError(Exception):
    """Exception raised for errors with collocation """
    pass


class CALIOP(CALIPSO):
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
        super().__init__("CALIOP", path)
        
        
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

    def get_subset(self, dataset, extent, return_coords=False):
        """
        Subset a particular CALIOP dataset to a geodetic extent.

        Parameters
        ----------
        dataset: str
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
        lons = self.get("Longitude")
        lats = self.get("Latitude")
        times = self.get_time()
        
        # Extract data
        data = self.get(dataset)

        if len(data.shape) != 2:
            raise ValueError("Dataset must be 2D")

        # Now apply subsetting
        subset_mask = (lons >= extent[0])*(lons <= extent[1])\
                        *(lats >= extent[2])*(lats <= extent[3])
        subset_data = data[subset_mask[:,0], :]
        
        if return_coords:
            return (subset_data, lons[subset_mask], lats[subset_mask],
                    times[subset_mask])
            
        else:
            return subset_data

    def get_interpolated_subset(self, dataset, extent, 
            ve1=0.0, ve2=40.0, vres=CALIOP_VERTICAL_RESOLUTION,
            return_coords=False):
        """
        Subset a particular CALIOP dataset to a geodetic extent and interpolate
        to a regular grid in the vertical.

        Parameters
        ----------
        dataset: str
            The name of the variable to subset
        extent: list
            List of minimum longitude, maximum longitude, minimum latitude,
            maximum latitude in degrees
        ve1: float (optional)
            The bottom of the interpolated grid in km
        ve2: float (optional)
            The top of the interpolated grid in km
        vres: float (optional)
            The vertical resolution to interpolate to, in m
        return_coords: bool (optional)
            Option to return longitudes, latitudes, times of subsetted profile
        
        Returns
        -------
        interpolated_subset: np.array
            Subsetted and interpolated data
        """
        subset, sub_lons, sub_lats, sub_times = self.get_subset(dataset,
                                                extent, return_coords=True)
        
        interpolated_subset = interpolate_caliop_profile(subset, ve1=ve1,
                                                    ve2=ve2, vres=vres)
        
        if return_coords:
            return interpolated_subset, sub_lons, sub_lats, sub_times
        else:
            return interpolated_subset
        
    def get_cloud_mask(self, backscatter_threshold=BACKSCATTER_THRESHOLD,
            width_threshold=WIDTH_THRESHOLD, 
            thickness_threshold=THICKNESS_THRESHOLD, area_threshold=10,
            return_backscatters=False, ve1=0.0, ve2=40.0,
            vres=CALIOP_VERTICAL_RESOLUTION, **kwargs):
        """
        Filters out noise in CALIOP L1 profiles based on thresholding the
        backscatter values. 

        Default parameters are as suggested by Iwabuchi et al. (2012)
        
        Parameters
        ----------
        b532: np.array
            CALIOP L1 attenuated backscatter at 532 nm (rows correspond to
            height)
        b1064: np.array
            CALIOP L1 attenuated backscatter at 1064 nm (rows correspond to
            height)
        backscatter_threshold: float (optional)
            Used to threshold the sum of the 532 and 1064 nm backscatters,
            default value from Iwabuchi et al. (2012)
        width_threshold: float (optional)
            The minimum width of a cloud in meters,
            default value corresponds to GOES-16 nadir pixel size
        thickness_threshold: float (optional)
            The minimum thickness of a cloud in meters
        area_threshold: float (optional)
            Minimum area in 'pixels'
        return_backscatters: bool (optional)
            Whether or not to return the backscatter profiles as well
        ve1: float (optional)
            The bottom of the interpolated grid in km
        ve2: float (optional)
            The top of the interpolated grid in km
        vres: float (optional)
            The vertical resolution to interpolate to, in m
        Returns
        -------
        mask: np.array
            Cloud mask
        """
        extent = kwargs.get("extent", self.get_extent())

        b532, lons, lats, times = self.get_interpolated_subset(
                                    "Total_Attenuated_Backscatter_532",
                                    extent, return_coords=True,
                                    ve1=ve1, ve2=ve2, vres=vres)
        b1064 = self.get_interpolated_subset("Attenuated_Backscatter_1064",
                                    extent, return_coords=False,
                                    ve1=ve1, ve2=ve2, vres=vres)

        cloud_mask = get_cloud_mask_from_backscatters(b532.T, b1064.T,
                            backscatter_threshold=backscatter_threshold,
                            width_threshold=width_threshold,
                            thickness_threshold=thickness_threshold,
                            area_threshold=area_threshold)

        if not return_backscatters:
            return cloud_mask 
        else:
            return cloud_mask, b532.T, b1064.T, lons, lats, times
    
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

        try:
            extinction_qcs = self.get("ExtinctionQC_532")
        except KeyError:
            extinction_qcs = None

        top_alts = self.get("Layer_Top_Altitude")

        # Find coordinates of endpoints of (sub-)trajectory
        # Column 0 contains locations corresponding to start of a 5km segment
        # Column 2 contains locations corresponding to end of a 5km segment
        if self.is_ascending():
            rows = np.where((lats[:,0] >= lat_min)*(lats[:,2] <= lat_max))[0]
        else:
            rows = np.where((lats[:,2] >= lat_min)*(lats[:,0] <= lat_max))[0]

        # Filter rows based on feature classification flags and extinction
        # quality control
        rows_to_keep = []
        for r in rows[~top_alts.mask[rows, 0]]:
            fcf = fcfs[r, 0]

            if extinction_qcs is not None:
                extinctionqc = extinction_qcs[r,0]
            else:
                extinctionqc = 16

            if not filter_cirrus_feature(fcf) or extinctionqc >= 16:
                continue 

            rows_to_keep.append(r)

        if not cirrus:
            rows_to_keep = rows[~np.isin(rows, rows_to_keep)]

        return rows_to_keep
    
    def get_backscatter_cmap(self):

        return loadcolormap(os.path.join(os.path.dirname(__file__), "assets",
                                        "calipso-backscatter.cmap"), "")

    def get_extent(self):

        lons = self.get("Longitude")
        lats = self.get("Latitude")

        lon_start = lons[0,0]
        lon_end = lons[-1,0]
        lat_start = lats[0,0]
        lat_end = lats[-1,0]

        if self.is_ascending():
            extent = [lon_end, lon_start, lat_start, lat_end]
        else:
            extent = [lon_start, lon_end, lat_end, lat_start]

        return extent
    

    def plot_backscatter(self, wavelength=532,
                        cloud_filter=False, fig=None, ax=None, extent=None,
                        **kwargs):
        if wavelength not in [532, 1064]:
            raise ValueError("Choose one of 532, 1064 for wavelength")

        if extent is None:
            extent = self.get_extent()

        if cloud_filter:
            cloud_mask, b532, b1064, lons, lats, times = self.get_cloud_filter(extent=extent,
                                                                            return_backscatters=True, **kwargs)

            if wavelength == 532:
                data_itp = b532 * cloud_mask.T
            else:
                data_itp = b1064 * cloud_mask.T
        else:
            if wavelength == 532:
                dataset = "Total_Attenuated_Backscatter_532"
            else:
                dataset = "Attenuated_Backscatter_1064"

            data, lons, lats, times = subset_caliop_profile(self, dataset,
                                                            extent, return_coords=True)

            data_itp = interpolate_caliop_profile(data, ve1=kwargs.get("min_alt", 0.0)*1000,
                                                        ve2=kwargs.get("max_alt", 40.)*1000)

        if fig is None and ax is None:
            fig, ax = plt.subplots(dpi=300, figsize=(15, 10))

        reverse = kwargs.get("reverse", False)
        if reverse:
            plot_caliop_profile_direct(fig, ax, lons[::-1], lats[::-1], times[::-1], data_itp.T[:,::-1], **kwargs)
        else:
            plot_caliop_profile_direct(fig, ax, lons, lats, times, data_itp.T, **kwargs)
        plt.close()
        return fig

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

    
def interpolate_caliop_profile(data, lidar_alts=CALIOP_ALTITUDES,
                        ve1=np.min(CALIOP_ALTITUDES),
                        ve2=np.max(CALIOP_ALTITUDES),
                        vres=CALIOP_VERTICAL_RESOLUTION):
    """
    Interpolates a CALIOP L1 dataset to a regular grid.

    Parameters
    ----------
    data: np.array
        Data to interpolate. Rows should correspond to height.
    lidar_alts: np.array (optional)
        The altitude (MSL) corresponding to the rows of 'data', in km
    ve1: float (optional)
        Vertical coordinate of bottom of interpolated grid in km
    ve2: float (optional)
        Vertical coordinate of top of interpolated grid in km
    vres: float (optional)
        The vertical resolution to interpolate to, in meters

    Returns
    -------
    interpolated: np.array
        Interpolated data
    """

    # Bounds for horizontal dimension, derived from data
    e1 = 0
    e2 = data.shape[0]
    
    X = np.arange(e1, e2, dtype=np.float32)

    # Multiply by a 1000 to convert to meters
    Y = np.meshgrid(1000*lidar_alts, X)[0].astype(np.float32)
    data = np.array(np.ma.masked_equal(data,  -9999).astype(np.float32))

    # Multiply by a 1000 to convert to meters
    interpolated = interp2d_12(data.astype(np.float32), 
                       X, Y,
                       float(e1), float(e2), int(e2 - e1), 
                       ve2 * 1000,
                       ve1 * 1000,
                       int(1000 * (ve2 - ve1) / vres))
    
    return interpolated
    

def get_cloud_mask_from_backscatters(b532, b1064, 
            backscatter_threshold=BACKSCATTER_THRESHOLD,
            width_threshold=WIDTH_THRESHOLD, 
            thickness_threshold=THICKNESS_THRESHOLD, area_threshold=10):
    """
    Filters out noise in CALIOP L1 profiles based on thresholding the
    backscatter values. 

    Default parameters are as suggested by Iwabuchi et al. (2012)
    
    Parameters
    ----------
    b532: np.array
        CALIOP L1 attenuated backscatter at 532 nm (rows correspond to height)
    b1064: np.array
        CALIOP L1 attenuated backscatter at 1064 nm (rows correspond to height)
    backscatter_threshold: float (optional)
        Used to threshold the sum of the 532 and 1064 nm backscatters,
        default value from Iwabuchi et al. (2012)
    width_threshold: float (optional)
        The minimum width of a cloud in meters,
        default value corresponds to GOES-16 nadir pixel size
    thickness_threshold: float (optional)
        The minimum thickness of a cloud in meters
    area_threshold: float (optional)
        Minimum area in 'pixels'
    Returns
    -------
    mask: np.array
        Cloud mask
    """
    
    mask = b532 + b1064 >= backscatter_threshold

    # Returns connected components in the mask
    ccl = skimage.measure.label(mask, connectivity=1)
    
    # Loop though connected components
    to_remove = []
    for c in skimage.measure.regionprops(ccl):
        
        # Bounding box of connected component
        box = c.bbox

        # Get width and thickness in meters
        width = (box[3] - box[1]) * CALIOP_HORIZONTAL_RESOLUTION
        thickness = (box[2] - box[0]) * CALIOP_VERTICAL_RESOLUTION
        
        # Check threshold conditions
        conds = [thickness < thickness_threshold,
                width < width_threshold,
                c.area < area_threshold]

        if np.any(conds):
            to_remove.append(c.label)
    
    negative_mask = np.isin(ccl, to_remove)
    updated_mask = mask * (~negative_mask)
    
    return updated_mask