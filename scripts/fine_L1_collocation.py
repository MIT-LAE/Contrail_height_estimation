#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH -J fine_collocation_r
"""
Use this script to do a fine collocation between CALIOP L1 products
and contrail detections.
"""

import os, sys, glob
import numpy as np, pandas as pd, datetime as dt
import multiprocessing 


from CAP.caliop import *
from CAP.geometry import * 
from CAP.collocation  import *
from CAP.abi import *
from CAP.utils import *

from contrails.meteorology.era5 import *

SAVE_DIR = "/home/vmeijer/contrail-height-estimation/data/"
SUFFIX = "_fine_collocation.parquet"


def era5_wrapper(time):

    if (time > dt.datetime(2021, 12, 31)) and (time <= dt.datetime(2022, 10, 20)):
        return xr.open_dataset("/net/d15/data/vmeijer/ERA5/" + time.strftime("%Y/%Y_%m_%d.nc"))
    elif (time <= dt.datetime(2023, 1, 1)) & (time >= dt.datetime(2022, 10, 21)):
        return xr.open_dataset("/net/d15/data/vmeijer/ERA5/" + time.strftime("%Y/%Y_%m_%d.grib"))
    else:
        return get_ERA5_data(time)


def main(input_path, save_path):


    if input_path.endswith(".pkl"):
        input_df = pd.read_pickle(input_path)
    else:
        input_df = pd.read_parquet(input_path)

    if "caliop_path" not in input_df.columns:
        print(f"No coarse collocations found for {input_path}.")
        return
    
    input_df['caliop_path'] = input_df['caliop_path'].str.replace("d13", "d15").str.replace("data/CALIPSO/", "")

    try:
        df = fine_collocation(input_df, get_mask, era5_wrapper, verbose=True)
    except FileNotFoundError as e:
        print(e)
        return

    if df is not None:
        df.to_pickle(save_path)
    return 


    
def get_mask(time, conus=False):

    if conus:
        df = pd.read_csv("/net/d13/data/vmeijer/data/orthographic_detections_goes16/ABI-L2-MCMIPC/" + time.strftime("%Y/%j/%H/%Y%m%d_%H_%M.csv"))
    else:
        df = pd.read_csv("/net/d13/data/vmeijer/data/orthographic_detections_goes16/ABI-L2-MCMIPF/" + time.strftime("%Y/%j/%H/%Y%m%d_%H_%M.csv"))


    mask = np.zeros((2000, 3000))
    mask[df.row.values, df.col.values] = 1
    return mask

if __name__ == "__main__":
    from multiprocessing import Pool
    
    paths = ["/home/vmeijer/contrail-height-estimation/data/CAL_LID_L1-Standard-V4-10.2018-08-08T18-58-40ZD_Subset_coarse_collocation.parquet"]
    
    save_paths = []
    for p in paths:
        save_paths.append(os.path.join(SAVE_DIR,
                    os.path.basename(p).replace(".hdf", SUFFIX)))

    if sys.argv[-1] == "DEBUG":
        for p, s in zip(paths, save_paths):
            main(p, s)
    else:
        n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

        pool = multiprocessing.Pool(n_cpus)

        print(f"Running {__file__} in parallel using {n_cpus} CPUs")

        pool.starmap(main, zip(paths, save_paths))
        pool.close()