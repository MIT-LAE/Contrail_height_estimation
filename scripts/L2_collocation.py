#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH -J L2_collocation
"""
Use this script to do a collocation between CALIOP L2 5 km cloud-layer cirrus data
and GOES-16 ABI data.
"""

import os, sys, glob
import numpy as np, pandas as pd, datetime as dt
import multiprocessing 

sys.path.append("/home/vmeijer/contrail-height-estimation/src/")

from caliop import *
from geometry import * 
from collocation  import *
from abi import *
from utils import *

from contrails.meteorology.era5 import *

SAVE_DIR = "/home/vmeijer/contrail-height-estimation/data/L2/"



def main(input_path, save_path):

    if os.path.exists(save_path):
        print(f"Already done for {input_path}, result at {save_path}")
        return
    try:
        coarse_df = coarse_L2_collocation(input_path, verbose=True)

        if "caliop_path" in coarse_df.columns:
            df = fine_L2_collocation(coarse_df, get_ERA5_data, verbose=True)
            df.to_pickle(save_path)

            return 
        else:
            print(f"No cirrus layers in {input_path}")
            return
        
    except Exception as e:
        print(f"Failed for {input_path} with {str(e)}")
        return

if __name__ == "__main__":
    paths = np.sort(glob.glob("/net/d13/data/vmeijer/data/CALIPSO/CALIOP_L2/CAL_LID_L2*.hdf"))
    save_paths = np.array([SAVE_DIR + os.path.basename(p).replace(".hdf",".pkl") for p in paths])

    if sys.argv[-1] == "DEBUG":
        for p, s in zip(paths, save_paths):
            main(p, s)
    else:
        n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

        pool = multiprocessing.Pool(n_cpus)

        print(f"Running {__file__} in parallel using {n_cpus} CPUs")

        pool.starmap(main, zip(paths, save_paths))
        pool.close()