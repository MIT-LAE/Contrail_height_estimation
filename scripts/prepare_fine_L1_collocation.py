#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH -J fine_prep
"""
Use this to prepare the fine L1 collocation.
Preparation entails:
* Segmenting each CALIOP overpass according to the ABI swath row
* Finding the closest GOES-16 ABI product to each segment
* Discarding segments without any coarse collocation
"""

import os, sys, glob
import numpy as np, pandas as pd, datetime as dt
import multiprocessing 
from collections import defaultdict

from contrails.satellites.goes.abi import *
from contrails.meteorology.era5 import *

from src.caliop import *
from src.geometry import * 
from src.collocation import *

from contrails.satellites.goes.utils import *


# Time when Full Disk GOES-16 product refresh rate changed from 15 to 10 minutes
TRANSITION_TIME = dt.datetime(2019, 4, 2)
GEODETIC_PATH = "/home/vmeijer/temp/contrails/contrails/contrails/satellites/assets/"
SAVE_DIR = "/home/vmeijer/height_estimation/data/prep_fine_L1/"
COARSE_COLLOCATION_PATH = "/home/vmeijer/height_estimation/data/coarse_L1/"


def main(p):
    try:
        prepare_fine_L1_collocation(p)
        return
    except Exception as e:
        print(f"Failed for {p} with {str(e)}")
        return

def prepare_fine_L1_collocation(path):
    
    
    coarse = pd.read_csv(COARSE_COLLOCATION_PATH \
                + os.path.basename(path).replace(".hdf", ".csv"))
    
    if "result" in coarse.columns:
        print(f"No coarse collocations found for {os.path.basename(path)}, skipping")
        return
    
    save_path = SAVE_DIR + os.path.basename(path).replace(".hdf", ".csv")
    
    if os.path.exists(save_path):
        print(f"Found L1 collocation prep result for {os.path.basename(path)}, skipping")
        return

    print(f"Started L1 collocation prep for {os.path.basename(path)}")

    ca = CALIOP(path)
    lons = ca.get("Longitude").data[:,0]
    lats = ca.get("Latitude").data[:,0]
    times = ca.get_time()[:,0]
    
    if lats[-1] > lats[0]:
        ascending = True
    else:
        ascending = False
        
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
            sub_coarse = coarse[(coarse.lat >= segment[0])*(coarse.lat <= segment[1])]
            idx = (lats >= segment[0])*(lats <= segment[1])
        else:
            sub_coarse = coarse[(coarse.lat >= segment[1])*(coarse.lat <= segment[0])]
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
        closest_product, product_time = find_closest_product(halfway_time,
                                             ABI_FD_row, ABI_FD_col)
        
        # Store result
        data["product"].append(closest_product)
        data["product_time"].append(product_time)
        data["segment_start_lat"].append(segment[0])
        data["segment_end_lat"].append(segment[1])
        
        
    df = pd.DataFrame(data)
    df.to_csv(save_path)
    print(f"Finished L1 collocation prep for {os.path.basename(path)}")


if __name__ == "__main__":
    #paths = np.sort(glob.glob("/net/d13/data/vmeijer/data/CALIPSO/CALIOP_L1/*.hdf")
    from multiprocessing import Pool
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