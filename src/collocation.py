from skimage.measure import label, regionprops
import numpy as np

import datetime as dt
import scipy.interpolate
from contrails.meteorology.advection import *
from contrails.satellites.goes.abi import * 

# TODO: Remove me when open sourcing
ASSET_PATH = "/home/vmeijer/contrails/contrails/satellites/assets/"

COLUMNS = ["Layer_Top_Altitude", "Layer_Base_Altitude", "Feature_Classification_Flags", "ExtinctionQC_532",
"Opacity_Flag", "Feature_Optical_Depth_532", "Feature_Optical_Depth_Uncertainty_532",
 "Ice_Water_Path", "Ice_Water_Path_Uncertainty", "Snow_Ice_Surface_Type"]

def apply_cloud_filter(b532, b1064, backscatter_threshold=0.003,
                             width_threshold=2000, thickness_threshold=100, area_threshold=10):
    """
    Filters out noise in CALIOP L1 profiles based on thresholding the backscatter
    values. 
    
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
    ccl = label(mask, connectivity=1)
    
    to_remove = []
    for c in regionprops(ccl):
        
        box = c.bbox

        # Horizontal resolution is 333 m
        width = (box[3] - box[1])*333

        # Vertical resolution is 30 m
        thickness = (box[2]-box[0])*30
        
        if thickness < thickness_threshold or width < width_threshold or c.area < area_threshold:
            to_remove.append(c.label)
    
    negative_mask = np.isin(ccl, to_remove)
    updated_mask = mask*(~negative_mask)
    
    return updated_mask 


def geometricaltitude2pressure(lon, lat, time, h, ds):
    """
    Convert geometric altitude by using the geopotential altitude.
    
    Parameters
    ----------
    lon: float
        Longitude in degrees
    lat: float
        Latitude in degrees
    time: dt.datetime
        Time in UTC
    h: float
        Geometric altitude in meters
    ds: xr.Dataset
        ERA5 weather dataset
        
    Returns
    -------
    pressure: float
        Pressure in hPa
    """
    
    gph_values = ds.z.interp(longitude=lon, latitude=lat, time=time,
                         method="linear").values/9.81

    itp = scipy.interpolate.interp1d(gph_values, ds.isobaricInhPa.values,
                                        fill_value="extrapolate")
    
    pressure = itp(h)
    
    return pressure


def map_heights_to_pressure(lons, lats, times, heights, ds):
    
    indices = np.where(~np.isnan(heights))[0]
    
    pressures = np.nan*np.zeros_like(heights)
    
    previous_height = 0.
    previous_pressure = 0.
    for i in indices:
        current_height = heights[i]
        if current_height == previous_height:
            pressures[i] = previous_pressure
        else:
            pressures[i] = geometricaltitude2pressure(lons[i], lats[i], times[i], current_height, ds)
        
        previous_pressure = pressures[i]
        previous_height = heights[i]
        
    return pressures
            
        
def get_interpolated_winds(lons, lats, times, heights, pressures, weather):
    
    
    era5_lons = weather.longitude.values.astype(np.float64)
    era5_lats = weather.latitude.values.astype(np.float64)
    era5_pressures = weather.isobaricInhPa.values.astype(np.float64)
    era5_times = weather.time.values
    u = weather.u.values.astype(np.float64)
    v = weather.v.values.astype(np.float64)

    us = np.nan*np.zeros_like(pressures)
    vs = np.nan*np.zeros_like(pressures)
    indices = np.where(~np.isnan(heights))[0]
    for i in indices:
        u_itp, v_itp = interpolate_winds(lons[i], lats[i], pressures[i], times[i], u, v, era5_lons, era5_lats,
                                        era5_pressures, era5_times)

        us[i] = u_itp
        vs[i] = v_itp

    
    return us, vs

    
def get_advected_positions(lons, lats, times, heights, pressures, weather, adv_time):
    
    
    u = weather.u.values.astype(np.float64)
    v = weather.v.values.astype(np.float64)
    longitudes = weather.longitude.values.astype(np.float64)
    latitudes = weather.latitude.values.astype(np.float64)
    pressures_k = weather.isobaricInhPa.values.astype(np.float64)
    times_l = weather.time.values
    
    adv_lons = np.nan*np.zeros_like(pressures)
    adv_lats = np.nan*np.zeros_like(pressures)
    indices = np.where(~np.isnan(heights))[0]
    for i in indices:
        x0 = np.array([lons[i], lats[i]])
        Dt = [(adv_time - times[i]).total_seconds()]
        t_ode = np.hstack((np.array([0.0]), np.array(Dt)))
        sol = odeint(advection_rhs, x0, t_ode,
                    args=(times[i], pressures[i], u, v, longitudes, latitudes,
                    pressures_k, times_l))
        
        adv_lons[i] = sol[1,0]
        adv_lats[i] = sol[1,1]

    
    return adv_lons, adv_lats
    
def round_conus_time(t):
    
    return round_time(t, minute_res=5)
    
def round_time(t, minute_res=10):
    
    minutes = int(np.round(t.minute/minute_res)*minute_res) % 60
    
    if minutes == 0 and t.minute > (60-minute_res/2):
        return dt.datetime(t.year, t.month, t.day, t.hour+1, minutes)
    else:
        return dt.datetime(t.year, t.month, t.day, t.hour, minutes)
    
    
def floor_time(t, minute_res=10):
    
    minutes = int(np.floor(t.minute/minute_res)*minute_res) % 60
    
    if minutes == 0 and t.minute > (60-minute_res/2):
        return dt.datetime(t.year, t.month, t.day, t.hour+1, minutes)
    else:
        return dt.datetime(t.year, t.month, t.day, t.hour, minutes)
    
def get_pixel_time_parameters(nc):
    scan_mode = int(nc.attrs["timeline_id"].split(" ")[-1])
    scene_id = nc.attrs["scene_id"]
    
    # If conus, check whether it's the first or second CONUS scan during the full disk scan
    if scene_id == "CONUS":
        
        if round_conus_time(np.datetime64(nc.attrs["time_coverage_end"]).astype(dt.datetime)).minute % 10 == 0:
            scene_id += "2"
        else:
            scene_id += "1"
    return scan_mode, scene_id

def get_pixel_time_LUT(scan_mode):
    
    return xr.open_dataset(ASSET_PATH + f"mode{scan_mode}.nc")

def find_closest_product(caliop_time, ABI_FD_row, ABI_FD_col):
    
    
    
    ABI_CONUS_row = ABI_FD_row - CONUS_FIRST_ROW
    ABI_CONUS_col = ABI_FD_col - CONUS_FIRST_COL
    
    
    # Mode 3, 3 CONUS scans for each 15 minute Full disk scan
    if caliop_time <= TRANSITION_TIME:
        
        # Find start of full disk scan
        fd_time = floor_time(caliop_time, minute_res=15)
        
        # We work with relative times
        rel_time = np.timedelta64(caliop_time - fd_time)
    
        # Get pixel time look up table
        ds = get_pixel_time_LUT(3)
        
        # Get delta_times for each region
        # Col/row reversal is on purpose due to LUT conventions
        fd_t = ds["FD_pixel_times"][ABI_FD_col, ABI_FD_row].values
        
        # If out of domain, set to very large values
        if ABI_CONUS_row >= 1500 or ABI_CONUS_col >= 2500:
            c1_t = np.timedelta64(100000, 's')
            c2_t = np.timedelta64(100000, 's')
            c3_t = np.timedelta64(100000, 's')
            
        else:
            c1_t = ds["CONUS1_pixel_times"][ABI_CONUS_col, ABI_CONUS_row].values + np.timedelta64(int(1000*(2*60+21.7)),'ms')
            c2_t = ds["CONUS2_pixel_times"][ABI_CONUS_col, ABI_CONUS_row].values + np.timedelta64(int(1000*(7*60+21.7)),'ms')
            c3_t = ds["CONUS3_pixel_times"][ABI_CONUS_col, ABI_CONUS_row].values + np.timedelta64(int(1000*(12*60+21.7)),'ms')
        
        fd_dt = (rel_time - fd_t)/np.timedelta64(1,'s')
        c1_dt = (rel_time - c1_t)/np.timedelta64(1,'s')
        c2_dt = (rel_time - c2_t)/np.timedelta64(1,'s')
        c3_dt = (rel_time - c3_t)/np.timedelta64(1,'s')
        
        closest_product = ["FD", "CONUS1", "CONUS2" ,"CONUS3"][np.argmin(np.abs([fd_dt, c1_dt, c2_dt, c3_dt]))]
        
        start_time = fd_time
        if "CONUS" in closest_product:
            start_time += dt.timedelta(minutes=5*(int(closest_product[-1])-1))
        return closest_product, start_time
        
    # Mode 6, 2 CONUS scans for each 10 minute Full disk scan
    else:
        # Find start of full disk scan
        fd_time = floor_time(caliop_time, minute_res=10)
        
        # We work with relative times
        rel_time = np.timedelta64(caliop_time - fd_time)
    
        # Get pixel time look up table
        ds = get_pixel_time_LUT(6)
        
        # Get delta_times for each region
        fd_t = ds["FD_pixel_times"][ABI_FD_col, ABI_FD_row].values
        
        # If out of domain, set to large values
        if ABI_CONUS_row >= 1500 or ABI_CONUS_col >= 2500:
            c1_t = np.timedelta64(100000, 's')
            c2_t = np.timedelta64(100000, 's')
            
        else:
            c1_t = ds["CONUS1_pixel_times"][ABI_CONUS_col, ABI_CONUS_row].values + np.timedelta64(int(1000*(2*60+21.7)),'ms')
            c2_t = ds["CONUS2_pixel_times"][ABI_CONUS_col, ABI_CONUS_row].values + np.timedelta64(int(1000*(7*60+21.7)),'ms')
            
        fd_dt = (rel_time - fd_t)/np.timedelta64(1,'s')
        c1_dt = (rel_time - c1_t)/np.timedelta64(1,'s')
        c2_dt = (rel_time - c2_t)/np.timedelta64(1,'s')

        closest_product = ["FD", "CONUS1", "CONUS2"][np.argmin(np.abs([fd_dt, c1_dt, c2_dt]))]
        
        start_time = fd_time
        if "CONUS" in closest_product:
            start_time += dt.timedelta(minutes=5*(int(closest_product[-1])-1))
        return closest_product, start_time
        
        
        
def label_scan_rows(lons, lats, boundaries=np.array([0, 229, 483, 737, 991, 1245]), conus_first_row=422,
                   conus_first_col=902):
    x_caliop, y_caliop = geodetic2ABI(lons, lats)
    rows, cols = get_ABI_grid_locations(x_caliop,y_caliop)
    rows -= conus_first_row
    cols -= conus_first_col

    scan_rows = np.argmin(np.maximum(rows[:,np.newaxis] - boundaries[np.newaxis,:],
                          -10e3*(rows[:,np.newaxis] - boundaries[np.newaxis,:])), axis=1)
    
    return scan_rows

def get_ABI_grid_locations(x, y, dx=5.5998564e-05, dy=5.5998564e-05):
    cols = np.floor(x/dx).astype(np.int64) + 2712
    rows = -np.floor(y/dy).astype(np.int64) + 2711
    return rows, cols

def segment_caliop_product(lons, lats, times):
    
    # Closest CONUS product time to each profile in the L1 product
    conus_times = [round_conus_time(t) for t in times]
    
    # Determine whether orbit is ascending (i.e. moving northward) or descending (moving southward)
    # We can use this method only because we know that every latitude is positive due to the
    # online subsetting we've done.
    ascending = (lats[0] - lats[-1]) < 0
    
    # Unique CONUS files to work with
    unique_times = np.unique(conus_times)
    
    scan_rows = label_scan_rows(lons, lats)
    
    # Loop through unique times and check whether the CONUS pixel times corresponding to the 
    # part of the CALIOP orbit are sufficiently close. If not, split the track
    
    segment_boundaries = []
    for unique_time in unique_times:
        
        idx = np.array(conus_times) == unique_time
        sub_lats = lats[idx]
        
        # Scan rows
        unique_rows = np.unique(scan_rows[idx])
                                
        # If true, we need to split this up
        if len(unique_rows) > 1:
            
            previous_lat = sub_lats[0]
            
            # if ascending, need to swap row order
            if ascending:
                unique_rows = unique_rows[::-1]
            
            for i in range(1, len(unique_rows)):
                
                boundary_lat = sub_lats[np.where(scan_rows[idx] == unique_rows[i])[0][0]]
        
                segment_boundaries.append((previous_lat, boundary_lat))
                previous_lat = boundary_lat

            segment_boundaries.append((previous_lat, sub_lats[-1]))
        else:
            segment_boundaries.append((sub_lats[0], sub_lats[-1]))
            
            
    return segment_boundaries

def extract_heights(profile_ids, cloud_mask, lidar_alts):
    """
    Extract top and bottom height of the clouds within cloud_mask at the columns
    given within profile_id.
    
    Parameters
    ----------
    profile_ids: np.array
        Columns to extract height data for
    cloud_mask: np.array
        Cloud mask
    lidar_alts: np.array
        Lidar altitudes in km, corresponding to rows of cloud_mask
        
    Returns
    -------
    top_heights: np.array
        Top height of cloud in m
    bot_heights: np.array
        Bottom height of cloud in m
    """
    
    subset = cloud_mask[:, profile_ids]
    top_idx = np.argmax(subset, axis=0)
    bot_idx = []

    for (idx, ID) in zip(top_idx, profile_ids):

        bot_idx.append(idx + np.argmax(1-cloud_mask[idx:, ID]))
        
    bot_idx = np.array(bot_idx)

    top_heights = lidar_alts[top_idx.astype(np.int32)]
    bot_heights = lidar_alts[bot_idx]
    
    return 1000*top_heights, 1000*bot_heights
    