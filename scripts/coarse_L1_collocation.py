#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH -J coarse_collocation
"""
Use this script to do a coarse collocation between CALIOP L1 products
and contrail detections.
"""

import os, sys, glob
import numpy as np, pandas as pd, datetime as dt
import multiprocessing 
from collections import defaultdict

from src.caliop import *
from src.geometry import * 

from contrails.satellites.goes.utils import *

# Time when Full Disk GOES-16 product refresh rate changed from 15 to 10 minutes
TRANSITION_TIME = dt.datetime(2019, 4, 2)
GEODETIC_PATH = "/home/vmeijer/temp/contrails/contrails/contrails/satellites/assets/"
SAVE_DIR = "/home/vmeijer/height_estimation/data/coarse_L1/"

def get_mask(time, conus=False):

    if conus:
        path = "/net/d13/data/vmeijer/data/orthographic_detections_goes16/" \
                + "ABI-L2-MCMIPC" + time.strftime("/%Y/%j/%H/%Y%m%d_%H_%M.csv")
        try:
            df = pd.read_csv(path) 
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No detection found at {path}")
            
        mask = np.zeros((2000, 3000))
        mask[df.row.values, df.col.values] = 1
        return mask
    else:
        df = pd.read_csv("/home/vmeijer/covid19/data/predictions_wo_sf/" + time.strftime('%Y%m%d.csv'))
        df.datetime = pd.to_datetime(df.datetime)
        df = df[df.datetime == time]
        mask = np.zeros((2000, 3000))
        mask[df.x.values, df.y.values] = 1
        return mask


def main(path):

    try:
        coarse_collocation(path, conus=True)
        return 
    except Exception as e:
        print(f"Failed for {path} with {str(e)}")
        return


def coarse_collocation(path, threshold_dist=50.0, conus=False):
    """
    Perform a "coarse" collocation of a CALIOP L1 file at 'path' 
    by checking whether any contrails have been detected on GOES-16 images
    that were captured during the CALIPSO overpass. 

    Parameters
    ----------
    path: str
        Location of CALIOP L1 file
    threshold_dist: float
        Maximum crosstrack distance for looking for contrails, in km
    """

    save_path = SAVE_DIR + os.path.basename(path).replace(".hdf",".csv")
    if os.path.exists(save_path):
         print(f"Found collocation result for {save_path}, skipping")
         return

    print(f"Started coarse collocation for {os.path.basename(path)}")
    
    ca = CALIOP(path)
    lons = ca.get("Longitude")
    lats = ca.get("Latitude")
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
    orbit = GroundTrack(lons, lats)
    
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
        
        df.to_csv(save_path)

        print(f"Finished coarse collocation for {os.path.basename(path)}, found {n_collocated} candidate pixels")
    else:
        print(f"Finished coarse collocation for {os.path.basename(path)}, no contrails found")
        pd.DataFrame({"result":["no collocations found"]}).to_csv(save_path)

if __name__ == "__main__":
    #paths = np.load("/home/vmeijer/height_estimation/notebooks/nighttime_files.npy")
    paths = np.sort(glob.glob("/net/d13/data/vmeijer/data/CALIPSO/CALIOP_L1/CAL_LID_L1-Standard-V4-11.2021*.hdf"))
    #paths = np.load("/home/vmeijer/height_estimation/notebooks/todo_2021.npy", allow_pickle=True)
    if sys.argv[-1] == "DEBUG":
        for p in paths:
            main(p)
    else:
        pool = multiprocessing.Pool(os.cpu_count())
        pool.map(main, paths)
        pool.close()