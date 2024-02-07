import numpy as np, datetime as dt
import xarray as xr
import cartopy.crs as ccrs

from .constants import GRS80_PARAMS
from .utils import floor_time, get_numpy_asset, get_netcdf_asset, get_ortho_ids



GOES16_PARAMS = {"h": 35786.0234375*1000., "lon_0": -75.2}

# Location of the CONUS product upper left corner in the Full Disk ABI grid
# Obtained from page 26 in 
# https://www.goes-r.gov/products/docs/PUG-L2+-vol3.pdf
# PUG Volume 3: Level 1b Products
# DCN 7035538, Revision H.1
CONUS_FIRST_COL = 902
CONUS_FIRST_ROW = 422

# Extent and projection for the orthographic projection centered at CONUS
# introduced in Kulik (2019): https://dspace.mit.edu/handle/1721.1/124179
ORTHO_EXTENT = [-2373970.2054802035,3638399.906890716, -2388159.0372323524,
                 1620933.528091551]
ORTHO_PROJ = ccrs.Orthographic(central_latitude=39.8283,
                                 central_longitude=-98.5795)

# Amount of rows and columns in the orthographic projection centered at CONUS
# introduced in Kulik (2019): https://dspace.mit.edu/handle/1721.1/124179
# see page 18
N_ORTHO_ROWS = 2000
N_ORTHO_COLS = 3000

# Table A1. from Carr et al. (2020)
# https://www.mdpi.com/2072-4292/12/22/3779
# contains the time offsets of the ABI bands in second
# i.e. [TIME_OFFSET_CHANNEL_1, TIME_OFFSET_CHANNEL_2, ...,
# TIME_OFFSET_CHANNEL_16]
GOES16_BAND_OFFSETS = [0.179, -0.055, 0.402, 0.642, -0.359, -0.642,	0.535,
            0.267, 0.000, -0.267, -0.535, -0.542, 0.551, 0.319, -0.256, 0.579]
                

def ABI2geodetic(x, y, nc=None):
    """
    Projects ABI fixed grid coordinates to GRS80 geodetic coordinates.
    Based on algorithm in section 5.1.2.8.1 in GOES-16 Product Definition and
    User's Guide, Version 3: Level 1b product.

    Parameters
    ----------
    x: float
        ABI fixed grid x-coordinate, radians
    y: float
        ABI fixed grid y-coordinate, radians
    nc: xr.Dataset (optional)
        ABI-L2-MCMIPF product

    Returns
    -------
    lon: float
        GRS80 geodetic longitude, degrees
    lat: float
        GRS80 geodetic latitude, degrees
    """

    if nc is not None:
        proj_info = nc['goes_imager_projection']
        
        lon_origin = proj_info.longitude_of_projection_origin
        H = proj_info.perspective_point_height + proj_info.semi_major_axis
        r_eq = proj_info.semi_major_axis
        r_pol = proj_info.semi_minor_axis
    else:
        # Values for `lon_origin` and `H` are from the reference mentioned
        # in the function docstring
        lon_origin = -75.0 # degrees
        r_eq = GRS80_PARAMS["a"] # meters
        r_pol = GRS80_PARAMS["b"] # meters
        H  = 42164160.0 # meters
    
    a = np.sin(x)**2 + np.cos(x)**2 * (np.cos(y)**2 \
            + (r_eq**2 / r_pol**2) * np.sin(y)**2)
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - r_eq**2
    r_s = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    s_x = r_s * np.cos(x) * np.cos(y)
    s_y = -r_s * np.sin(x)
    s_z = r_s * np.cos(x) * np.sin(y)

    lat = np.degrees(np.arctan((r_eq**2 / r_pol**2) \
                                    * s_z / (np.sqrt((H - s_x)**2 + s_y**2))))
    lon = lon_origin - np.degrees(np.arctan(s_y / (H - s_x)))
    
    return lon, lat

def geodetic2ABI(lon, lat, nc=None):
    """
    Projects GRS80 geodetic coordinates to  ABI fixed grid coordinates.
    Based on algorithm in section 5.1.2.8.1 in GOES-16 Product Definition and
    User's Guide, Version 3: Level 1b product.

    Parameters
    ----------
    lon: float
        GRS80 geodetic longitude, degrees
    lat: float
        GRS80 geodetic latitude, degrees
    nc: xr.Dataset (optional)
        ABI-L2-MCMIPF product

    Returns
    -------
    x: float
        ABI fixed grid x-coordinate, radians
    y: float
        ABI fixed grid y-coordinate, radians
    """
    # Value for eccentricity `e` is from the reference mentioned in the
    # docstring
    if nc is not None:
        proj_info = nc['goes_imager_projection']
        
        lon_origin = proj_info.longitude_of_projection_origin
        H = proj_info.perspective_point_height+proj_info.semi_major_axis
        r_eq = proj_info.semi_major_axis
        r_pol = proj_info.semi_minor_axis
        e = 0.0818191910435
    else:
        # Values for `lon_origin` and `H` are from the reference mentioned
        # in the function docstring
        lon_origin = -75.0 # degrees
        r_eq = GRS80_PARAMS["a"] # meters
        r_pol = GRS80_PARAMS["b"] # meters
        H  = 42164160.0 # meters
        e = 0.0818191910435
    
    phi_c = np.arctan((r_pol**2/r_eq**2)*np.tan(np.radians(lat)))
    r_c = r_pol/(np.sqrt(1-e**2 * np.cos(phi_c)**2))
    
    s_x = H - r_c*np.cos(phi_c)*np.cos(np.radians(lon-lon_origin))
    s_y = - r_c*np.cos(phi_c)*np.sin(np.radians(lon-lon_origin))
    s_z = r_c*np.sin(phi_c)

    y = np.arctan(s_z/s_x)
    x = np.arcsin(-s_y/(np.sqrt(s_x**2 + s_y**2 + s_z**2)))

    if np.any((H*(H-s_x)) < s_y**2 + (r_eq**2/r_pol**2)*s_z**2):
        raise ValueError("Geodetic coordinates not visible from satellite")

    return x, y

def map_geodetic_extent_to_ABI(extent, conus=False):
    """
    Utility function to map a geodetic extent to ABI product rows and columns

    Parameters
    ----------
    extent: list
        Geodetic extent [lon_min, lon_max, lat_min, lat_max]
    conus: bool
        Whether the extent should be mapped to CONUS product or Full-disk
        rows and columns

    Returns
    -------
    ABI_extent: list
        ABI extent [col_min, col_max, row_min, row_max]
    """
    x_ul, y_ul = geodetic2ABI(extent[0], extent[3])
    x_ll, y_ll = geodetic2ABI(extent[0], extent[2])

    x_ur, y_ur = geodetic2ABI(extent[1], extent[3])
    x_lr, y_lr = geodetic2ABI(extent[1], extent[2])

    x = [x_ul, x_ll, x_ur, x_lr]
    y = [y_ul, y_ll, y_ur, y_lr]
    
    r_min, c_min = get_ABI_grid_locations(min(x), max(y))
    r_max, c_max = get_ABI_grid_locations(max(x), min(y))
    
    if conus:
        r_min -= CONUS_FIRST_ROW
        r_max -= CONUS_FIRST_ROW
        c_min -= CONUS_FIRST_COL
        c_max -= CONUS_FIRST_COL
    
    return [max(c_min,0), max(c_max,0), max(r_min,0), max(r_max,0)]


def get_ABI_grid_locations(x, y, dx=56e-6, dy=56e-6):
    """
    Finds rows and columns in ABI full disk 2 km grid
    given `x` and `y` scan angles.
    The values `dx` and `dy` are obtained from table 5.1.2.4 in the GOES-16
    Product Definition and User's Guide, Version 3: Level 1b product.
    # DCN 7035538, Revision H.1
    Parameters
    ----------
    x: float
        ABI fixed grid x-coordinate, radians
    y: float
        ABI fixed grid y-coordinate, radians
    dx: float, optional
        ABI fixed grid x-coordinate resolution, radians
    dy: float, optional
        ABI fixed grid y-coordinate resolution, radians
    Returns
    -------
    rows: np.array
        Rows in ABI grid corresponding to `x` and `y`
    cols: np.array
        Columns in ABI grid corresponding to `x` and `y`
    """
    cols = np.floor(x/dx).astype(np.int64) + N_ABI_FD_COLS // 2
    
    # To understand the - 1, refer to figure 5.1.2.6-2 
    # in the GOES-16
    # Product Definition and User's Guide, Version 3: Level 1b product.
    # DCN 7035538, Revision H.1
    rows = -np.floor(y/dy).astype(np.int64) + (N_ABI_FD_ROWS // 2) - 1
    return rows, cols


def get_pixel_times(scan_mode, band, region="FD"):
    """
    For a particular scan mode, band and region, returns the time at which
    pixels within the ABI-L2-MCMIP product were scanned (relative to the start
    of scan).

    Parameters
    ----------
    scan_mode: int
        The scan mode
    band: int
        The ABI band (1-16)
    region: str (optional)
        The scanning region
    
    Returns
    -------
    times: np.array
        The GOES-16 ABI-L2-MCMIP pixel times
    """
    if band not in list(range(1,17)):
        raise ValueError("Band should be between 1 and 16")
    if scan_mode not in list(range(1,7)):
        raise ValueError("Scan mode should be between 1 and 6")
    
    # Load NetCDF file containing pixel scan times
    # these files were found in the Supplementary Materials of 
    # Carr et al. (2020):
    # https://www.mdpi.com/2072-4292/12/22/3779#app1-remotesensing-12-03779
    ds = get_netcdf_asset(f"mode{scan_mode}")

    times = ds[region +"_pixel_times"].values

    if "CONUS" in region:
        abi_ids = get_ortho_ids()
        rows, cols = np.unravel_index(abi_ids.flatten(), (5424, 5424)) 

        # Convert rows cols to those relevant for the CONUS files 
        x_map = get_numpy_asset("x_map")
        y_map = get_numpy_asset("y_map")
        
        cols_c = x_map[cols]
        rows_c = y_map[rows]
        times = times.T[rows_c, cols_c].reshape((2000, 3000))

    # Add offset to account for band
    times += np.timedelta64(int(1000*GOES16_BAND_OFFSETS[band-1]), "ms")
    return times


def get_scan_start_time(goes_time, scan_mode, product):
    """
    For a particular scan mode and product, returns the time at which the
    scan started.

    Parameters
    ----------
    goes_time: datetime
        The GOES-16 product time. For full-disk, this is a round time of 10
        or 15 minutes (e.g. 10:00, 10:15, 10:30, 10:45). For CONUS,
        this is a round time of 5 minutes (e.g. 10:00, 10:05, 10:10, 10:15)
    scan_mode: int
        The scan mode
    product: str
        The ABI product, either "FD" or "CONUS{1,2,3}" with the latter number
        indicating which of the three CONUS products is being used. CONUS1
        is the first CONUS product obtained during a particular FD scan, etc.
    
    Returns
    -------
    start_time: datetime
        The time at which the scan started
    """
    
    # Scan mode 3 corresponds to 1 Full disk product every 15 minutes
    if scan_mode == 3:
        product_time = floor_time(g oes_time, minute_res=15)
        if "CONUS" in product:
            
            conus_n = int(product[-1])
            
            # 21 second delta is based on looking at the observation
            # start times for GOES-16 ABI-L2-MCMIPC products when scan mode 3
            # was active. These start times are found in the filenames. 
            # They tend to fluctuate by a few seconds with a 21 second mean.
            # We add 5 minutes for every CONUS product after the first
            offset = dt.timedelta(minutes=(2 + (conus_n-1)*5),
                        seconds=21)
            return product_time + offset
            
        else:
            # 40 second delta is based on looking at the observation
            # start times for GOES-16 ABI-L2-MCMIPF products when scan mode 3
            # was still active. These start times are found in the filenames. 
            # They tend to fluctuate by < 5 seconds with a 40 second mean
            return product_time + dt.timedelta(seconds=40)
     
    # Scan mode 6
    else:
        product_time = floor_time(goes_time, minute_res=10)
        if "CONUS" in product:
            conus_n = int(product[-1])

            # 17 second delta is based on looking at the observation
            # start times for GOES-16 ABI-L2-MCMIPC products when scan mode 6
            # is active. These start times are found in the filenames. 
            # They tend to fluctuate by a few seconds with a 17 second mean
            # See https://www.ospo.noaa.gov/Operations/GOES/east/g16meso.html
            # for the up-to-date scheduling, and an explanation for these
            # fluctuations
            # We add 5 minutes for every CONUS product after the first
            return product_time + dt.timedelta(minutes=(1+(conus_n-1)*5),
                        seconds=17)
            
        else:
            # 20 second delta is based on looking at the observation
            # start times for GOES-16 ABI-L2-MCMIPF products when scan mode 6
            # is active. These start times are found in the filenames. 
            # They tend to fluctuate by a few seconds with a 20 second mean
            # See https://www.ospo.noaa.gov/Operations/GOES/east/g16meso.html
            # for the up-to-date scheduling, and an explanation for these
            # fluctuations
            return product_time + dt.timedelta(seconds=20)
        