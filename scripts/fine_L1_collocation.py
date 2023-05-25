#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH -J fine_collocation
"""
Use this script to do a fine collocation between CALIOP L1 products
and contrail detections.
"""

import os, sys, glob
import numpy as np, pandas as pd, datetime as dt
import multiprocessing 
from collections import defaultdict
from contrails.meteorology.era5 import *
from src.caliop import *
from src.geometry import * 
from src.collocation import *

from contrails.satellites.goes.utils import * 

from scipy.ndimage import distance_transform_edt

# Time when Full Disk GOES-16 product refresh rate changed from 15 to 10 minutes
TRANSITION_TIME = dt.datetime(2019, 4, 2)
GEODETIC_PATH = "/home/vmeijer/temp/contrails/contrails/contrails/satellites/assets/"
SAVE_DIR = "/home/vmeijer/height_estimation/data/fine_L1/"
PREPARATION_PATH = "/home/vmeijer/height_estimation/data/prep_fine_L1/"


def main(p):

    try:
        execute_fine_L1_collocation(p)
        return
    except Exception as e:
        print(f"Failed for {p} with {str(e)}")
        return

def execute_fine_L1_collocation(L1_path):


    save_path = SAVE_DIR + os.path.basename(L1_path).replace(".hdf", ".csv")
    
    if os.path.exists(save_path):
        print(f"Found fine L1 collocation for {os.path.basename(L1_path)}, skipping")
        return 

    try:
        # Dataframe containing results from preparation step
        prep_df = pd.read_csv(PREPARATION_PATH + os.path.basename(L1_path).replace(".hdf",".csv"))
    except FileNotFoundError:
        print(f"No fine collocation preparation file found for {os.path.basename(L1_path)}")
        return

    print(f"Starting fine L1 collocation for {os.path.basename(L1_path)}")
    
    goes_lons = get_lons()
    goes_lats = get_lats()

    ca = CALIOP(L1_path)
    profile_lons = ca.get("Longitude").data[:,0]
    profile_lats = ca.get("Latitude").data[:,0]
    profile_ids = np.arange(len(profile_lons))
    profile_times = ca.get_time()[:,0]

    if profile_lats[-1] > profile_lats[0]:
        ascending = True
    else:
        ascending = False


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

        subset_idx = (profile_lats >= lat_min)*(profile_lats <= lat_max)

        # Subset caliop data
        extent = [-135, -45, lat_min, lat_max]
        b532, lons, lats, times = subset_caliop_profile(ca, "Total_Attenuated_Backscatter_532",
                                                extent, return_coords=True)
        extent[0] = lons.min()
        extent[1] = lons.max()
        b532 = interpolate_caliop_profile(b532, lidar_alts=np.array(ca.Lidar_Data_Altitudes), ve1=8*1000., ve2=15.*1000)
        b1064 = subset_caliop_profile(ca, "Attenuated_Backscatter_1064", extent)
        b1064 = interpolate_caliop_profile(b1064, lidar_alts=np.array(ca.Lidar_Data_Altitudes), ve1=8*1000., ve2=15.*1000)

        cloud_mask = np.ma.masked_invalid(apply_cloud_filter(b532.T, b1064.T, backscatter_threshold=0.007,
                                                        thickness_threshold=200, width_threshold=4000))

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
        pixel_times = get_pixel_times(scan_mode, 11, region=goes_product) + np.datetime64(scan_start_time)


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
        mask = get_mask(goes_time, conus=True)
        #edt = distance_transform_edt(mask)

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
            
            # # Get relevant edt values
            # edts = []
            # for ID in caliop_ids[matched]:
            #     rows, cols = np.where(orthographic_ABI_ids == ID)
            #     edts.append(np.mean(edt[rows, cols]))
            # data["contrail_edt"] = edts
            
    df = pd.DataFrame(data)
    df.to_csv(save_path)
    print(f"Finished fine L1 collocation for {os.path.basename(L1_path)}")
    return
    

def get_mask(time, conus=False):
    
    if conus:
        path = "/net/d13/data/vmeijer/data/orthographic_detections_goes16/" \
            + "ABI-L2-MCMIPC" + time.strftime("/%Y/%j/%H/%Y%m%d_%H_%M.csv")
        df = pd.read_csv(path) 
        mask = np.zeros((2000, 3000))
        mask[df.row.values.astype(np.int64), df.col.values.astype(np.int64)] = 1
        return mask
    else:
        df = pd.read_csv("/home/vmeijer/covid19/data/predictions_wo_sf/" + time.strftime('%Y%m%d.csv'))
        df.datetime = pd.to_datetime(df.datetime)
        df = df[df.datetime == time]
        mask = np.zeros((2000, 3000))
        mask[df.x.values, df.y.values] = 1
        return mask 


def get_scan_start_time(goes_time, scan_mode, product):
    
    
    if scan_mode == 3:
        product_time = floor_time(goes_time, minute_res=15)
        if "CONUS" in product:
            conus_n = int(product[-1])
            return product_time + dt.timedelta(minutes=(2+(conus_n-1)*5),seconds=21, milliseconds=700)
            
        else:
            return product_time + dt.timedelta(seconds=40, milliseconds=700)
     
    # Scan mode 6
    else:
        product_time = floor_time(goes_time, minute_res=10)
        if "CONUS" in product:
            conus_n = int(product[-1])
            return product_time + dt.timedelta(minutes=(1+(conus_n-1)*5),seconds=18, milliseconds=300)
            
        else:
            return product_time + dt.timedelta(seconds=21, milliseconds=600)
        
if __name__ == "__main__":
    from multiprocessing import Pool
    import sys
    root = "/net/d13/data/vmeijer/data/CALIPSO/CALIOP_L1/"
    #paths = [root+os.path.basename(p) for p in np.load("/home/vmeijer/height_estimation/notebooks/todo.npy", allow_pickle=True)]

    paths = np.sort(glob.glob("/net/d13/data/vmeijer/data/CALIPSO/CALIOP_L1/CAL_LID_L1-Standard-V4-11.2021*.hdf"))

    # test_path = "/net/d13/data/vmeijer/data/CALIPSO/CALIOP_L1/CAL_LID_L1-Standard-V4-10.2018-08-08T18-58-40ZD_Subset.hdf"
    # execute_fine_L1_collocation(test_path)
    # for file in paths:
    #     execute_fine_L1_collocation(file)
    if len(sys.argv) > 1:
        if sys.argv[-1] == "DEBUG":
            print("DEBUG")
            for p in paths:
                main(p)
    else:
        pool = Pool(os.cpu_count())
        pool.map(main, paths)
        pool.close()
