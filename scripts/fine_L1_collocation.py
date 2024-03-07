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

import os
import sys
import glob
import datetime as dt
import multiprocessing 

import numpy as np
import pandas as pd
import click

from CAP.collocation import fine_collocation
from contrails.meteorology.era5 import get_ERA5_data
from utils import process_multiple

INPUT_SUFFIX = "_coarse_collocation.parquet"
OUTPUT_SUFFIX = "_fine_collocation.parquet"

def era5_wrapper(time):

    if (time > dt.datetime(2021, 12, 31)) and (time <= dt.datetime(2022, 10, 20)):
        return xr.open_dataset("/net/d15/data/vmeijer/ERA5/" + time.strftime("%Y/%Y_%m_%d.nc"))
    elif (time <= dt.datetime(2023, 1, 1)) & (time >= dt.datetime(2022, 10, 21)):
        return xr.open_dataset("/net/d15/data/vmeijer/ERA5/" + time.strftime("%Y/%Y_%m_%d.grib"))
    else:
        return get_ERA5_data(time)


def process_file(input_path, save_path):

    print(f"Processing {input_path}")
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
        print(f"Found {len(df)} fine collocations for {input_path}")
        df.to_parquet(save_path)
    return 
    
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

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("save_dir", type=click.Path())
@click.option("--debug", is_flag=True, default=False)
def main(input_path, save_dir, debug):
    process_multiple(process_file, input_path, save_dir, INPUT_SUFFIX,
                        OUTPUT_SUFFIX, parallel=not debug)



if __name__ == "__main__":
    main()