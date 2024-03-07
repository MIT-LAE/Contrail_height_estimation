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

import os
import sys
import glob
import multiprocessing 

import click
import numpy as np
import pandas as pd
import datetime as dt

from CAP.caliop import *
from CAP.geometry import * 
from CAP.collocation  import *
from CAP.abi import *
from CAP.utils import *

SUFFIX = "_coarse_collocation.parquet"

def get_mask(time, conus=False):

    if conus or time > dt.datetime(2021, 12, 31):

        suffix = "F"
        if conus:
            suffix = "C"
        path = "/net/d13/data/vmeijer/data/orthographic_detections_goes16/" \
                    + "ABI-L2-MCMIP" + suffix + time.strftime("/%Y/%j/%H/%Y%m%d_%H_%M.csv")
        try:
            df = pd.read_csv(path) 
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No detection found at {path}")
            
        mask = np.zeros((2000, 3000))
        mask[df.row.values, df.col.values] = 1
        return mask
    else:
        df = pd.read_csv("/home/vmeijer/covid19/data/predictions_wo_sf/"\
                            + time.strftime('%Y%m%d.csv'))
        df.datetime = pd.to_datetime(df.datetime)
        df = df[df.datetime == time]
        mask = np.zeros((2000, 3000))
        mask[df.x.values, df.y.values] = 1
        return mask

def process_file(input_path, save_path):

    if os.path.exists(save_path):
        print(f"Already done for {input_path}, result at {save_path}")
        return
    try:
        df = coarse_collocation(input_path, get_mask, conus=True, verbose=True)
        df.to_parquet(save_path)
        return 
    except Exception as e:
        print(f"Failed for {input_path} with {str(e)}")
        return

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("save_dir", type=click.Path())
@click.option("--debug", is_flag=True, default=False)
def main(input_path, save_dir, debug):
    if os.path.isdir(input_path):
        glob_path = os.path.join(input_path, "*.hdf")
        paths = glob.glob(glob_path)
    else:
        paths = pd.read_csv(input_path, header=None)[0].values

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_paths = []
    for p in paths:
        save_paths.append(os.path.join(save_dir,
                    os.path.basename(p).replace(".hdf", SUFFIX)))
        
    if debug:
        for p, s in zip(paths, save_paths):
            process_file(p, s)
    else:
        n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

        pool = multiprocessing.Pool(n_cpus)

        print(f"Running {__file__} in parallel using {n_cpus} CPUs")

        pool.starmap(process_file,
                        zip(paths, save_paths))
        pool.close()
    

if __name__ == "__main__":
    main()