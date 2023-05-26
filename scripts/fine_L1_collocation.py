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

sys.path.append(os.path.dirname(__file__) + "../src/")

from caliop import *
from geometry import * 
from collocation  import *
from abi import *
from utils import *

from contrails.meteorology.era5 import *

SAVE_DIR = "/home/vmeijer/contrail-height-estimation/data/fine/"

def main(input_path, save_path):

    

    if os.path.exists(save_path):
        print(f"Already done for {input_path}, result at {save_path}")
        return

    input_df = pd.read_pickle(input_path)

    if "caliop_path" not in input_df.columns:
        print(f"No coarse collocations found for {input_path}.")
        return
    
    try:
        df = fine_collocation(input_df, get_mask, get_ERA5_data, verbose=True)
        df.to_pickle(save_path)
        return 
    except Exception as e:
        print(f"Failed for {input_path} with {str(e)}")
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

if __name__ == "__main__":
    from multiprocessing import Pool
    
    paths = np.sort(glob.glob("/home/vmeijer/contrail-height-estimation/data/coarse/*.pkl"))

    save_paths = np.array([SAVE_DIR + os.path.basename(p) for p in paths])

    if sys.argv[-1] == "DEBUG":
        for p, s in zip(paths, save_paths):
            main(p, s)
    else:
        n_cpus = os.environ.get("SLURM_CPUS_PER_TASK", 1)

        pool = multiprocessing.Pool(n_cpus)

        print(f"Running {__file__} in parallel using {n_cpus} CPUs")

        pool.starmap(main, zip(paths, save_paths))
        pool.close()