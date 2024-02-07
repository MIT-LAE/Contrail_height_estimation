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


import sys
sys.path.append("/home/vmeijer/contrail-height-estimation/src/")

from collocation import *
from contrails.satellites.goes.utils import *


# Time when Full Disk GOES-16 product refresh rate changed from 15 to 10 minutes
TRANSITION_TIME = dt.datetime(2019, 4, 2)

SAVE_DIR = "/home/vmeijer/contrail-height-estimation/data/prep_fine_L1/"
COARSE_COLLOCATION_PATH = "/home/vmeijer/contrail-height-estimation/data/coarse/"


def main(p):
    try:

        coarse_df = pd.read_pickle("/home/vmeijer/contrail-height-estimation/data/coarse/" + os.path.basename(p).replace(".hdf", ".pkl"))

        df = prepare_fine_collocation(coarse_df, verbose=True)
        if df is not None:
            df.to_pickle(SAVE_DIR + os.path.basename(p).replace(".hdf", ".pkl"))
        return
    except Exception as e:
        print(f"Failed for {p} with {str(e)}")
        return


if __name__ == "__main__":

    from multiprocessing import Pool
    paths = np.sort(glob.glob("/net/d15/data/vmeijer/CALIOP_L1/CAL_LID_L1-*2023*.hdf"))

    if len(sys.argv) > 1:
        if sys.argv[-1] == "DEBUG":
            print("DEBUG")
            for p in paths:
                main(p)
    else:
        pool = Pool(os.cpu_count())
        pool.map(main, paths)
        pool.close()