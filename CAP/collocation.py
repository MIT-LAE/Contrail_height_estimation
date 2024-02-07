import os
from collections import defaultdict
import datetime as dt

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import scipy.interpolate
import scipy.constants

from .caliop import CALIOP
from .geometry import parallax_correction_vicente_backward
from .advection import get_interpolated_winds, get_advected_positions
from .abi import (get_ABI_grid_locations, geodetic2ABI, CONUS_FIRST_COL,
                CONUS_FIRST_ROW, get_scan_start_time, get_pixel_times,
                map_geodetic_extent_to_ABI, TRANSITION_TIME)
from .utils import (get_lons, get_lats, get_ortho_ids, get_netcdf_asset,
                    floor_time, round_conus_time)
from .vertical_feature_mask import get_cirrus_fcf_integers


# See https://www.eoportal.org/satellite-missions/calipso#
# Refers to L1b product resolution
CALIOP_HORIZONTAL_RESOLUTION = 333 # m
CALIOP_VERTICAL_RESOLUTION = 30 # m

# See Iwabuchi et al. (2012), section 3.3:
# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JD017020
BACKSCATTER_THRESHOLD = 0.003 # km^-1 sr^-1
WIDTH_THRESHOLD = 1000 # meter
THICKNESS_THRESHOLD = 60 # meter

# Altitude thresholds used in paper
MINIMUM_ALTITUDE = 8.0 # km
MAXIMUM_ALTITUDE = 15.0 # km

# This is the geodetic extent used to subset CALIOP data in the CALIPSO
# subsetting web app https://subset.larc.nasa.gov/calipso/login.php
# It covers the contrail detection domain from Meijer et al. (2022)
# The order is [min_lon, max_lon, min_lat, max_lat]
CALIOP_SUBSET_EXTENT = [-135, -45, 10, 55]


COLUMNS = ["Layer_Top_Altitude", "Layer_Base_Altitude",
            "Feature_Classification_Flags", "ExtinctionQC_532",
            "Opacity_Flag", "Feature_Optical_Depth_532",
            "Feature_Optical_Depth_Uncertainty_532", "Ice_Water_Path",
            "Ice_Water_Path_Uncertainty", "Snow_Ice_Surface_Type"]

def apply_cloud_filter(b532, b1064, 
            backscatter_threshold=BACKSCATTER_THRESHOLD
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
    ccl = label(mask, connectivity=1)
    
    # Loop though connected components
    to_remove = []
    for c in regionprops(ccl):
        
        # Bounding box of connected component
        box = c.bbox

        # Get width and thickness in meters
        width = (box[3] - box[1]) * CALIOP_HORIZONTAL_RESOLUTION
        thickness = (box[2]-box[0]) * CALIOP_VERTICAL_RESOLUTION
        
        # Check threshold conditions
        conds = [thickness < thickness_threshold,
                width < width_threshold,
                c.area < area_threshold]

        if np.any(conds):
            to_remove.append(c.label)
    
    negative_mask = np.isin(ccl, to_remove)
    updated_mask = mask * (~negative_mask)
    
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
    
    # ERA5 geopotential altitude is in m^2/s^2
    gph_values = ds.z.interp(longitude=lon, latitude=lat, time=time,
                         method="linear").values / scipy.constants.g

    itp = scipy.interpolate.interp1d(gph_values, ds.isobaricInhPa.values,
                                        fill_value="extrapolate")
    
    pressure = itp(h)
    
    return pressure


def map_heights_to_pressure(lons, lats, times, heights, ds):
    """
    Convert geometric altitude by using the geopotential altitude,
    but only do interpolation when necessary to avoid unnecessary
    computation.
    """
    
    indices = np.where(~np.isnan(heights))[0]
    
    pressures = np.nan*np.zeros_like(heights)
    
    previous_height = 0.
    previous_pressure = 0.
    for i in indices:
        current_height = heights[i]

        # Don't do anything if the height is the same as the previous
        # This implicitly assumes that the height variation is smaller
        # than the geopotential variation
        if current_height == previous_height:
            pressures[i] = previous_pressure
        else:
            pressures[i] = geometricaltitude2pressure(lons[i], lats[i],
                                                times[i], current_height, ds)
        
        previous_pressure = pressures[i]
        previous_height = heights[i]
        
    return pressures  


        
        
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
    
    # Loop through unique times and check whether the CONUS pixel times
    # corresponding to the part of the CALIOP orbit are sufficiently close. 
    # If not, split the track
    
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
                
                boundary_lat = sub_lats[np.where(scan_rows[idx] \
                                            == unique_rows[i])[0][0]]
        
                segment_boundaries.append((previous_lat, boundary_lat))
                previous_lat = boundary_lat

            segment_boundaries.append((previous_lat, sub_lats[-1]))
        else:
            segment_boundaries.append((sub_lats[0], sub_lats[-1]))
            
    return segment_boundaries

def extract_heights(profile_ids, cloud_mask, lidar_alts):
    """
    Extract top and bottom height of the clouds within cloud_mask at the
    columns given within profile_id.
    
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
    


def coarse_collocation(path, get_mask, threshold_dist=50.0, conus=False,
                            verbose=False):
    """
    Perform a "coarse" collocation of a CALIOP L1 file at 'path' 
    by checking whether any contrails have been detected on GOES-16 images
    that were captured during the CALIPSO overpass. 

    50 km is the default threshold distance used in the paper. Rationale:
    The maximum possible time between a contrail observed in a GOES-16 image
    and the time that CALIPSO is near it is about 15 minutes, since we use
    the average time for the CALIPSO overpass (which tends to be ~30 minutes
    long). Thus, assuming a max wind speed of 60 m/s, we find an advection time
    of about 50 km in these 15 minutes.

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
    
    # Minutes between each ABI-L2-MCMIP product 
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
        mask_time = dt.datetime(t.year, t.month, t.day, t.hour,
                        int(np.round(t.minute/minute_interval)*minute_interval))
    except ValueError:
        # Doing it this way takes care of cornercase where the
        # next hour is on the next day
        next_h = t + dt.timedelta(hours=1)
        mask_time = dt.datetime(next_h.year, next_h.month, next_h.day,
                                    next_h.hour, 0)
        
    found = False
    og_mask_time = mask_time - dt.timedelta(minutes=30)
    max_delta = 12
    delta = 0
    times_tried = []
    while not found and delta < max_delta:
        mask_time = og_mask_time + dt.timedelta(minutes=delta * 5)
        times_tried.append(mask_time)
        try:
            mask = get_mask(mask_time, conus=conus)
            found = True
        except FileNotFoundError:
            delta += 1
    
    if not found:
        raise FileNotFoundError("No detections at " \
                            + ", ".join([str(t) for t in times_tried]))
        
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
            print(f"Finished coarse collocation for {os.path.basename(path)},"\
                    + f" found {n_collocated} candidate pixels")

        return df 
    else:
        if verbose:
            print(f"Finished coarse collocation for {os.path.basename(path)},"\
                + " no contrails found")
        return pd.DataFrame({"result":["no collocations found"]})



def fine_collocation(coarse_df, get_mask, get_ERA5_data, verbose=False):

    L1_path = coarse_df.iloc[0].caliop_path

    prep_df = prepare_fine_collocation(coarse_df, verbose=verbose)
    if prep_df is None:
        print(f"No coarse collocations remain after preparation, skipping")
        return
    if verbose:
        print(f"Starting fine L1 collocation for {os.path.basename(L1_path)}")

    ca = CALIOP(L1_path)
    profile_lons = ca.get("Longitude").data[:,0]
    profile_lats = ca.get("Latitude").data[:,0]
    profile_ids = np.arange(len(profile_lons))

    ascending = profile_lats[-1] > profile_lats[0]

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
        extent = CALIOP_SUBSET_EXTENT[:2] + [lat_min, lat_max]

        cloud_mask, b532, b1064, lons, lats, times = ca.get_cloud_filter(
                                            extent=extent,
                                            return_backscatters=True,
                                            min_alt=MINIMUM_ALTITUDE,
                                            max_alt=MAXIMUM_ALTITUDE)

        extent[0] = lons.min()
        extent[1] = lons.max()

        weather = get_ERA5_data(goes_time)
        
        heights = np.linspace(MINIMUM_ALTITUDE * 1000, MAXIMUM_ALTITUDE * 1000,
                    cloud_mask.shape[0])[::-1][np.argmax(cloud_mask, axis=0)]
        heights[heights == MAXIMUM_ALTITUDE] = np.nan

        ps = map_heights_to_pressure(lons, lats, times, heights, weather)

        if goes_time < TRANSITION_TIME:
            scan_mode = 3
        else:
            scan_mode = 6

        scan_start_time = get_scan_start_time(goes_time, scan_mode,
                                                goes_product)

        # Take GOES-16 ABI band 11 as representative
        # as it is used during the contrail detection process described
        # in Meijer et al. (2022)
        pixel_times = get_pixel_times(scan_mode, 11, region=goes_product) \
                        + np.datetime64(scan_start_time)


        if "CONUS" in goes_product:
            conus = True
        else:
            conus = False

        ABI_extent = map_geodetic_extent_to_ABI(extent, conus=conus)

        sub_times = np.sort(pixel_times[ABI_extent[2]:ABI_extent[3],
                                    ABI_extent[0]:ABI_extent[1]].flatten())

        # Pixel time advected to is the median time
        # the difference among these times is usually small (< 5 seconds)
        # but can have outliers if near the swath edges
        # hence take the 'median', not the average
        median_idx = len(sub_times)//2
        advection_time = pd.Timestamp(sub_times[median_idx]).to_pydatetime()

        indices = ~np.isnan(heights)

        # Do advection
        adv_lons, adv_lats = get_advected_positions(lons[indices],
                                                    lats[indices],
                                                    times[indices],
                                                    heights[indices],
                                                    ps[indices],
                                                    weather,
                                                    advection_time)

        subindices = ~(np.isnan(adv_lons)+np.isnan(adv_lats))
        adv_lons = adv_lons[subindices]
        adv_lats = adv_lats[subindices]

        try:
            # Invert parallax
            lons_c, lats_c = parallax_correction_vicente_backward(adv_lons,
                                                                 adv_lats,
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

        caliop_ids = N_ABI_FD_COLS * ABIgrid_rows + ABIgrid_cols
        matched = np.isin(caliop_ids, contrail_ids.astype(np.int64))

        # Store results
        n_matched = int(np.sum(matched).item())

        try:
            profile_ids[subset_idx][indices][subindices][matched]
        except IndexError:
            print(f"Failed for segment {i} in {L1_path}")
            continue

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

            matched_rows, matched_cols = np.unravel_index(caliop_ids[matched],
                                                (N_ABI_FD_ROWS, N_ABI_FD_ROWS))
            data["goes_ABI_row"].extend(list(matched_rows))
            data["goes_ABI_col"].extend(list(matched_cols))
            data["goes_product"].extend(n_matched*[goes_product])
            data["goes_time"].extend(n_matched*[goes_time])
            
            
    df = pd.DataFrame(data)
    if verbose:
        print(f"Finished fine L1 collocation for {os.path.basename(L1_path)}")
    return df

def prepare_fine_collocation(coarse_df, verbose=False, L2=False):
    
    if "result" in coarse_df.columns:
        print(f"No coarse collocations found, skipping")
        return

    if verbose:
        print("Started L1 collocation prep for "\
                + f"{coarse_df.iloc[0].caliop_path}")


    ca = CALIOP(coarse_df.iloc[0].caliop_path)
    if L2:
        lons = ca.get("Longitude")[:,1]
        lats = ca.get("Latitude")[:,1]
        times = ca.get_time()[:,1]
        
    else:
        lons = ca.get("Longitude").data[:,0]
        lats = ca.get("Latitude").data[:,0]
        times = ca.get_time()[:,0]

        mask = (lats >= CALIOP_SUBSET_EXTENT[2]) \
                * (lats <= CALIOP_SUBSET_EXTENT[3]) \
                * (lons >= CALIOP_SUBSET_EXTENT[0]) \
                * (lons <= CALIOP_SUBSET_EXTENT[1])
        lons = lons[mask]
        lats = lats[mask]
        times = times[mask]
    
    if len(lats) == 0:
        print(f"No coarse collocations found in domain, skipping")
        return

    # If last latitude is greater than first, orbit is ascending.
    # This only works for the subsetted CALIOP data we are using in this work
    # as we have the guarantee that the latitude is either monotonically
    # increasing or decreasing
    ascending = lats[-1] > lats[0]

    segments = segment_caliop_product(lons, lats, times)
    
    # To store results
    data = defaultdict(list)
    coarse_df = coarse_df.sort_values(by='lat')

    for segment in segments:
     
        # Check if there were any contrails nearby according to coarse
        # collocation
        if ascending:
            sub_coarse = coarse_df[(coarse_df.lat >= segment[0])\
                                    *(coarse_df.lat <= segment[1])]
            idx = (lats >= segment[0])*(lats <= segment[1])
        else:
            sub_coarse = coarse_df[(coarse_df.lat >= segment[1])\
                                    *(coarse_df.lat <= segment[0])]
            idx = (lats >= segment[1])*(lats <= segment[0])
            
        # If there were no contrails nearby, skip to next segment
        if len(sub_coarse) == 0:

            continue
            
        # Extract coordinates
        sub_lats = lats[idx]
        sub_lons = lons[idx]
        sub_times = times[idx]
        
        # Take halfway point
        halfway_lat = sub_lats[len(sub_lats) // 2]
        halfway_lon = sub_lons[len(sub_lats) // 2]
        halfway_time = sub_times[len(sub_lats) // 2]
        
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
        print("Finished L1 collocation prep for "\
                + f"{coarse_df.iloc[0].caliop_path}")
    return df


def coarse_L2_collocation(path, verbose=False):
    
    if verbose:
        print(f"Started coarse collocation for {os.path.basename(path)}")
        
    ca = CALIOP(path)
    
    # Figure out which L2 layers are for cirrus
    cirrus_ints = get_cirrus_fcf_integers()
    fcfs = ca.get("Feature_Classification_Flags")
    cirrus_mask = np.isin(fcfs, cirrus_ints)
    row_mask = cirrus_mask.sum(axis=1) > 0
    
    # Coordinates of middle of 5 km layer
    lats = ca.get("Latitude")[row_mask,1]
    lons = ca.get("Longitude")[row_mask,1]
    alts = ca.get("Layer_Top_Altitude")[row_mask,0]
    times = ca.get_time()[row_mask, 1]
    
    n_collocated = sum(row_mask)
    
    if n_collocated > 0:
        
        df = pd.DataFrame({'caliop_path' : n_collocated * [path],
                           'caliop_time' : times, 
                           'detection_time' : n_collocated * [""],
                            'lat': lats,
                            'lon': lons,
                             'height': alts})
        
        df["caliop_time"] = pd.to_datetime(df["caliop_time"])
        
        if verbose:
            print(f"Finished coarse collocation for {os.path.basename(path)},"\
                + f" found {n_collocated} candidate cirrus layers")

        return df 
    else:
        if verbose:
            print(f"Finished coarse collocation for {os.path.basename(path)},"\
                +" no cirrus found")
        return pd.DataFrame({"result":["no collocations found"]})
        
        
def fine_L2_collocation(coarse_df, get_ERA5_data, verbose=False):

    L2_path = coarse_df.iloc[0].caliop_path

    prep_df = prepare_fine_collocation(coarse_df, L2=True, verbose=verbose)
    if len(prep_df) == 0:
        print(f"No coarse collocations remain after preparation, skipping")
    if verbose:
        print(f"Starting fine L2 collocation for {os.path.basename(L2_path)}")

    ca = CALIOP(L2_path)
    lons = coarse_df.lon.values
    lats = coarse_df.lat.values
    alts = coarse_df.height.values
    times = np.array([pd.Timestamp(t).to_pydatetime() for \
                        t in coarse_df.caliop_time.values])

    profile_ids = np.arange(len(lons))
    
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

        subset_idx = (lats >= lat_min) * (lats <= lat_max)
        
        sub_lons = lons[subset_idx]
        sub_lats = lats[subset_idx]
        sub_times = times[subset_idx]

        # Convert km to m
        sub_heights = alts[subset_idx] * 1000
        
        weather = get_ERA5_data(goes_time)

     
        ps = map_heights_to_pressure(sub_lons, sub_lats, sub_times,
                                    sub_heights, weather)

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
        
        extent = [sub_lons.min(), sub_lons.max(), lat_min, lat_max]
        
        ABI_extent = map_geodetic_extent_to_ABI(extent, conus=conus)

        sub_pixel_times = np.sort(pixel_times[ABI_extent[2]:ABI_extent[3],
                                        ABI_extent[0]:ABI_extent[1]].flatten())

        median_idx = len(sub_pixel_times) // 2
        advection_time = pd.Timestamp(sub_pixel_times[median_idx]
                                        ).to_pydatetime()


        # Do advection
        adv_lons, adv_lats = get_advected_positions(sub_lons, sub_lats,
                                                    sub_times,
                                                    sub_heights,
                                                    ps, 
                                                    weather,
                                                    advection_time)

        subindices = ~(np.isnan(adv_lons)+np.isnan(adv_lats))
        adv_lons = adv_lons[subindices]
        adv_lats = adv_lats[subindices]

        try:
            # Invert parallax
            lons_c, lats_c = parallax_correction_vicente_backward(adv_lons,
                                                                adv_lats,
                                                    sub_heights[subindices])
        except ValueError:
            print("Parallax correction error")
            print(goes_time, L1_path)
            return


        x, y = geodetic2ABI(lons_c, lats_c)

        # Map to ABI coordinates
        ABIgrid_rows, ABIgrid_cols = get_ABI_grid_locations(x,y)
        
        orthographic_ABI_ids = get_ortho_ids()

        caliop_ids = N_ABI_FD_COLS * ABIgrid_rows + ABIgrid_cols

        n_matched = len(caliop_ids)
        
        if n_matched > 0:
        
            data["L1_file"].extend(n_matched*[os.path.basename(L2_path)])
            data["segment_start_lat"].extend(n_matched*[row.segment_start_lat])
            data["segment_end_lat"].extend(n_matched*[row.segment_end_lat])
            data["segment_start_idx"].extend( n_matched*[row.start_profile_id])
            data["segment_end_idx"].extend(n_matched*[row.end_profile_id])
            data["profile_id"].extend(list(profile_ids[subset_idx][subindices]
                                            ))
            data["caliop_lon"].extend(list(sub_lons[subindices]))
            data["caliop_lon_adv"].extend(list(adv_lons))
            data["caliop_lat"].extend(list(sub_lats[subindices]))
            data["caliop_lat_adv"].extend(list(adv_lats))
            data["caliop_top_height"].extend(list(sub_heights[subindices]))
            data["caliop_lon_adv_parallax"].extend(list(lons_c))
            data["caliop_lat_adv_parallax"].extend(list(lats_c))
            data["caliop_time"].extend(list(sub_times[subindices]))
            data["caliop_pressure_hpa"].extend(list(ps[subindices]))
            data["adv_time"].extend(n_matched*[advection_time])
            data["goes_ABI_id"].extend(list(caliop_ids))

            matched_rows, matched_cols = np.unravel_index(caliop_ids,
                                                (N_ABI_FD_ROWS, N_ABI_FD_ROWS))
            data["goes_ABI_row"].extend(list(matched_rows))
            data["goes_ABI_col"].extend(list(matched_cols))
            data["goes_product"].extend(n_matched*[goes_product])
            data["goes_time"].extend(n_matched*[goes_time])
            
            
    df = pd.DataFrame(data)

    if verbose:
        print(f"Finished fine L2 collocation for {os.path.basename(L2_path)}")
    return df