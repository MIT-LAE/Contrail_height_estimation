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

from CAP.collocation import coarse_collocation
from utils import process_multiple, get_mask

INPUT_SUFFIX = ".hdf"
OUTPUT_SUFFIX = "_coarse_collocation.parquet"

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
    process_multiple(process_file, input_path, save_dir, INPUT_SUFFIX,
                        OUTPUT_SUFFIX, parallel=not debug)

if __name__ == "__main__":
    main()