from skimage.measure import label, regionprops
import numpy as np, pandas as pd
import os
from collections import defaultdict

import datetime as dt
import scipy.interpolate
from caliop import *
from geometry import *

from contrails.meteorology.advection import *
from abi import *
from utils import *


# Time when Full Disk GOES-16 product refresh rate changed from 15 to 10 minutes
TRANSITION_TIME = dt.datetime(2019, 4, 2)

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

def find_closest_ABI_product(caliop_time, ABI_FD_row, ABI_FD_col):
    """
    Finds the GOES-16 ABI product for which the pixel capture time at
    the location indicated by 'ABI_FD_row' and 'ABI_FD_col' is closest
    to 'caliop_time'.

    Uses the pixel capture time estimates from Carr et al. (2020).

    Parameters
    ----------
    caliop_time : dt.datetime
        The CALIOP profile time in UTC
    ABI_FD_row : int
        The ABI Full disk row
    ABI_FD_col : int
        The ABI Full disk col

    Returns
    -------
    closest_product : str
        The found ABI product name
    start_time : dt.datetime
        The scan start time of the found ABI product
    """
    
    ABI_CONUS_row = ABI_FD_row - CONUS_FIRST_ROW
    ABI_CONUS_col = ABI_FD_col - CONUS_FIRST_COL
    
    
    # Mode 3, 3 CONUS scans for each 15 minute Full disk scan
    if caliop_time <= TRANSITION_TIME:
        
        # Find start of full disk scan
        fd_time = floor_time(caliop_time, minute_res=15)
        
        # We work with relative times
        rel_time = np.timedelta64(caliop_time - fd_time)
    
        # Get pixel time look up table
        ds = get_netcdf_asset('mode3')
        
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
        ds = get_netcdf_asset("mode6")
        
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
    """
    The orthographic projection used for the Meijer et al. (2022) contrail detections
    is covered by 6 ABI "scan rows". This function finds the scan row of each
    input coordinate.

    Parameters
    ----------
    lons : np.array
        Longitude, degrees
    lats : np.array
        Latitude, degrees
    boundaries : np.array (optional)
        The ABI fixed-grid rows of the scan row boundaries
    conus_first_row : int (optional)
        The first row of the ABI conus product (w.r.t the Full disk fixed-grid)
    conus_first_col : int (optional)
        The first column of the ABI conus product (w.r.t the Full disk fixed-grid)
    
    Returns
    -------
    scan_rows : np.array
        The identified scan rows
    """
    x_caliop, y_caliop = geodetic2ABI(lons, lats)
    rows, cols = get_ABI_grid_locations(x_caliop,y_caliop)
    rows -= conus_first_row
    cols -= conus_first_col

    scan_rows = np.argmin(np.maximum(rows[:,np.newaxis] - boundaries[np.newaxis,:],
                          -10e3*(rows[:,np.newaxis] - boundaries[np.newaxis,:])), axis=1)
    
    return scan_rows


def segment_caliop_product(lons, lats, times):
    """
    Divide a CALIOP curtain into segments where each segment
    has a different closest (temporally) GOES-16 ABI product and time.
    
    Parameters
    ----------
    lons : np.array
        Longitude of CALIPSO ground track, degrees
    lats : np.array
        Latitude of CALIPSO ground track, degrees
    times : np.array
        Times of CALIPSO ground track, UTC
    
    Returns
    -------
    segment_boundaries : List[Tuple[float]]
        List of the start and end latitudes of each segment
    """
    
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
    


def coarse_collocation(path, get_mask, threshold_dist=50.0, conus=False, verbose=False):
    """
    Perform a "coarse" collocation of a CALIOP L1 file at 'path' 
    by checking whether any contrails have been detected on GOES-16 images
    that were captured during the CALIPSO overpass. 

    Parameters
    ----------
    path:  str
        Location of CALIOP L1 file
    get_mask : function
        Function to use for obtaining contrail masks
    threshold_dist : float (optional)
        Maximum crosstrack distance for looking for contrails, in km
    conus : bool (optional)
        Whether or not to use CONUS data
    verbose : bool(optional)
        Control print statements

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing collocation results
    """

    if verbose:
        print(f"Started coarse collocation for {os.path.basename(path)}")
    
    ca = CALIOP(path)

    times = ca.get_time() 
    
    # Take approximate average time 
    t = times[len(times)//2,0]
    
    # Minutes between each ABI-L2-MCMIPF product 
    if conus:
        minute_interval = 5
    else:
        if t < TRANSITION_TIME:
            minute_interval = 15
        else:
            minute_interval = 10
    
    # Get longitudes, latitudes for pixels in the projection used for detection 
    goes_lon = get_lons()
    goes_lat = get_lats()
    
    # Parameterize CALIPSO ground track
    orbit = ca.get_ground_track()
    
    # Find all pixels within threshold_dist of the CALIPSO ground track 
    dists = np.abs(orbit.get_crosstrack_distance(goes_lon, goes_lat))
    collocation_mask = dists <= threshold_dist
    
    # Get mask for time 
    try:
        mask_time = dt.datetime(t.year, t.month, t.day, t.hour, int(np.round(t.minute/minute_interval)*minute_interval))
    except ValueError:
        # Doing it this way takes care of cornercase where the next hour is on the next day
        next_h = t + dt.timedelta(hours=1)
        mask_time = dt.datetime(next_h.year, next_h.month, next_h.day, next_h.hour, 0)
        
    # if not conus:
    #     try:
    #         try:
    #             mask = get_mask(mask_time, 
    #                         prediction_dir='/home/vmeijer/covid19/data/predictions_wo_sf/')
    #         except FileNotFoundError:
    #             path = "/net/d13/data/vmeijer/data/orthographic_detections_goes16/" \
    #             + "ABI-L2-MCMIPF" + mask_time.strftime("/%Y/%j/%H/%Y%m%d_%H_%M.csv")
    #             print(path)
    #             df = pd.read_csv(path) 
    #             mask = np.zeros((2000, 3000))
    #             mask[df.row.values.astype(np.int64), df.col.values.astype(np.int64)] = 1
                
    #     except FileNotFoundError:
    #         print(f"Did not find contrail mask corresponding to {os.path.basename(path)}")
    #         print("Failed to find " + mask_time.strftime('%Y-%m-%d %H:%M'))
    #         return
    # else:
    mask = get_mask(mask_time, conus=conus)

    collocated = (mask==1.0)*collocation_mask
    
    # If any contrail pixels found close to the CALIPSO ground track,
    # indicate successful collocation
    if np.sum(collocated) > 0:

        # Extract information on collocated contrails
        lat_collocated = goes_lat[collocated]
        lon_collocated = goes_lon[collocated]
        dists_collocated = dists[collocated]
        rows, cols = np.where(collocated)

        n_collocated = int(np.sum(collocated))

        df = pd.DataFrame({'caliop_path' : n_collocated * [path],
                           'caliop_mean_time' : n_collocated * [t],
                           'detection_time' : n_collocated * [mask_time],
                            'lat': lat_collocated,
                            'lon': lon_collocated,
                            'dist': dists_collocated,
                            'row': rows,
                            'col': cols})
        
        #df.to_csv(save_path)
        if verbose:
            print(f"Finished coarse collocation for {os.path.basename(path)}, found {n_collocated} candidate pixels")

        return df 
    else:
        if verbose:
            print(f"Finished coarse collocation for {os.path.basename(path)}, no contrails found")
        return pd.DataFrame({"result":["no collocations found"]})



def fine_collocation(coarse_df, get_mask, get_ERA5_data, verbose=False):

    L1_path = coarse_df.iloc[0].caliop_path

    prep_df = prepare_fine_collocation(coarse_df)
    
    if verbose:
        print(f"Starting fine L1 collocation for {os.path.basename(L1_path)}")

    ca = CALIOP(L1_path)
    profile_lons = ca.get("Longitude").data[:,0]
    profile_lats = ca.get("Latitude").data[:,0]
    profile_ids = np.arange(len(profile_lons))

    ascending = ca.is_ascending()

    # To store results
    data = defaultdict(list)

    for i in range(len(prep_df)):

        row = prep_df.iloc[i]
        goes_product = row["product"]
        goes_time = pd.to_datetime(row.product_time)

        if ascending:
            lat_min = row.segment_start_lat
            lat_max = row.segment_end_lat
        else:
            lat_min = row.segment_end_lat
            lat_max = row.segment_start_lat

        subset_idx = (profile_lats >= lat_min) * (profile_lats <= lat_max)

        # Subset caliop data
        extent = [-135, -45, lat_min, lat_max]

        # b532, lons, lats, times = subset_caliop_profile(ca, "Total_Attenuated_Backscatter_532",
        #                                         extent, return_coords=True)
        
        

        cloud_mask, b532, b1064, lons, lats, times = ca.get_cloud_filter(extent=extent,
                                            return_backscatters=True, min_alt=8, max_alt=15)

        extent[0] = lons.min()
        extent[1] = lons.max()


        weather = get_ERA5_data(goes_time)

        heights = np.linspace(8e3, 15e3, cloud_mask.shape[0])[::-1][np.argmax(cloud_mask, axis=0)]
        heights[heights == 15.0e3] = np.nan
        ps = map_heights_to_pressure(lons, lats, times, heights, weather)
        #us, vs = get_interpolated_winds(lons, lats, times, heights, ps, weather)

        if goes_time < TRANSITION_TIME:
            scan_mode = 3
        else:
            scan_mode = 6

        scan_start_time = get_scan_start_time(goes_time, scan_mode, goes_product)
        pixel_times = get_pixel_times(scan_mode, 11, region=goes_product) \
                        + np.datetime64(scan_start_time)


        if "CONUS" in goes_product:
            conus = True
        else:
            conus = False

        ABI_extent = map_geodetic_extent_to_ABI(extent, conus=conus)
        # domain_idx = (goes_lons >= extent[0])*(goes_lons <= extent[1])\
        #         *(goes_lats >= extent[2])*(goes_lats <= extent[3])
        sub_times = np.sort(pixel_times[ABI_extent[2]:ABI_extent[3],ABI_extent[0]:ABI_extent[1]].flatten())

        median_idx = len(sub_times)//2
        advection_time = pd.Timestamp(sub_times[median_idx]).to_pydatetime()

        indices = ~np.isnan(heights)

        # Do advection
        adv_lons, adv_lats = get_advected_positions(lons[indices], lats[indices], times[indices],
                                    heights[indices], ps[indices], weather, advection_time)
        subindices = ~(np.isnan(adv_lons)+np.isnan(adv_lats))
        adv_lons = adv_lons[subindices]
        adv_lats = adv_lats[subindices]

        try:
            # Invert parallax
            lons_c, lats_c = parallax_correction_vicente_backward(adv_lons, adv_lats,
                                                                    heights[indices][subindices])
        except ValueError:
            print("Parallax correction error")
            print(goes_time, L1_path)
            return


        x, y = geodetic2ABI(lons_c, lats_c)

        # Map to ABI coordinates
        ABIgrid_rows, ABIgrid_cols = get_ABI_grid_locations(x,y)
        
        # Check contrail matches
        mask = get_mask(goes_time, conus=conus)

        orthographic_ABI_ids = get_ortho_ids()
        contrail_ids = orthographic_ABI_ids[mask==1.0]

        caliop_ids = 5424*(ABIgrid_rows) + ABIgrid_cols
        matched = np.isin(caliop_ids, contrail_ids.astype(np.int64))

        # Store results
        n_matched = int(np.sum(matched).item())
        
        if n_matched > 0:
        
            data["L1_file"].extend(n_matched*[os.path.basename(L1_path)])
            data["segment_start_lat"].extend( n_matched*[row.segment_start_lat])
            data["segment_end_lat"].extend(n_matched*[row.segment_end_lat])
            data["segment_start_idx"].extend( n_matched*[row.start_profile_id])
            data["segment_end_idx"].extend( n_matched*[row.end_profile_id])
            data["profile_id"].extend(list(profile_ids[subset_idx][indices][subindices][matched]))
            data["caliop_lon"].extend(list(lons[indices][subindices][matched]))
            data["caliop_lon_adv"].extend(list(adv_lons[matched]))
            data["caliop_lat"].extend(list(lats[indices][subindices][matched]))
            data["caliop_lat_adv"].extend(list(adv_lats[matched]))
            data["caliop_top_height"].extend(list(heights[indices][subindices][matched]))
            data["caliop_lon_adv_parallax"].extend(list(lons_c[matched]))
            data["caliop_lat_adv_parallax"].extend(list(lats_c[matched]))
            data["caliop_time"].extend(list(times[indices][subindices][matched]))
            data["caliop_pressure_hpa"].extend(list(ps[indices][subindices][matched]))
            data["adv_time"].extend(n_matched*[advection_time])
            data["goes_ABI_id"].extend(list(caliop_ids[matched]))

            matched_rows, matched_cols = np.unravel_index(caliop_ids[matched], (5424, 5424))
            data["goes_ABI_row"].extend(list(matched_rows))
            data["goes_ABI_col"].extend(list(matched_cols))
            data["goes_product"].extend(n_matched*[goes_product])
            data["goes_time"].extend(n_matched*[goes_time])
            
            
    df = pd.DataFrame(data)
    # df.to_csv(save_path)
    if verbose:
        print(f"Finished fine L1 collocation for {os.path.basename(L1_path)}")
    return df


def prepare_fine_collocation(coarse_df, verbose=False):
    
    
    if "result" in coarse_df.columns:
        print(f"No coarse collocations found, skipping")
        return

    if verbose:
        print(f"Started L1 collocation prep for {coarse_df.iloc[0].caliop_path}")

    ca = CALIOP(coarse_df.iloc[0].caliop_path)
    lons = ca.get("Longitude").data[:,0]
    lats = ca.get("Latitude").data[:,0]
    times = ca.get_time()[:,0]
    
    ascending = ca.is_ascending()

    # Segment overpass
    try:
        segments = segment_caliop_product(lons, lats, times)
    except ValueError:
        return
    
    # To store results
    data = defaultdict(list)
    
    for segment in segments:

        # Check if there were any contrails nearby according to coarse collocation
        if ascending:
            sub_coarse = coarse_df[(coarse_df.lat >= segment[0])*(coarse_df.lat <= segment[1])]
            idx = (lats >= segment[0])*(lats <= segment[1])
        else:
            sub_coarse = coarse_df[(coarse_df.lat >= segment[1])*(coarse_df.lat <= segment[0])]
            idx = (lats >= segment[1])*(lats <= segment[0])
            
        # If there were no contrails nearby, skip to next segment
        if len(sub_coarse) == 0:
            continue
            
        # Extract coordinates
        sub_lats = lats[idx]
        sub_lons = lons[idx]
        sub_times = times[idx]
        
        # Take halfway point
        halfway_lat = sub_lats[len(sub_lats)//2]
        halfway_lon = sub_lons[len(sub_lats)//2]
        halfway_time = sub_times[len(sub_lats)//2]
        
        # Compute ABI grid location of halfway point
        x, y = geodetic2ABI(halfway_lon, halfway_lat)
        ABI_FD_row, ABI_FD_col = get_ABI_grid_locations(x, y)
        
        # Get closest product, and product time
        closest_product, product_time = find_closest_ABI_product(halfway_time,
                                             ABI_FD_row, ABI_FD_col)
        
        # Store result
        data["product"].append(closest_product)
        data["product_time"].append(product_time)
        data["segment_start_lat"].append(segment[0])
        data["segment_end_lat"].append(segment[1])
        data["start_profile_id"].append(np.where(idx)[0][0])
        data["end_profile_id"].append(np.where(idx)[0][-1])
        
        
    df = pd.DataFrame(data)
    
    if verbose:
        print(f"Finished L1 collocation prep for {df.iloc[0].caliop_path}")
    return df

