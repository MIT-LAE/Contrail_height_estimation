#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH -J L2_collocation
"""
Use this script to do a collocation between CALIOP L2 5 km cloud-layer cirrus
and GOES-16 ABI data.
"""
import os
import sys
import glob

import click
import numpy as np
import pandas as pd
import datetime as dt

from CAP.caliop import CALIOP
from CAP.collocation  import fine_L2_collocation, coarse_L2_collocation
from contrails.meteorology.era5 import get_ERA5_data
from utils import process_multiple

INPUT_SUFFIX = ".hdf"
OUTPUT_SUFFIX = "_L2_collocation.parquet"


def process_file(input_path, save_path):

    if os.path.exists(save_path):
        print(f"Already done for {input_path}, result at {save_path}")
        return
    try:
        coarse_df = coarse_L2_collocation(input_path, verbose=True)

        if "caliop_path" in coarse_df.columns:
            df = fine_L2_collocation(coarse_df, get_ERA5_data, verbose=True)
            df.to_parquet(save_path)

            return 
        else:
            print(f"No cirrus layers in {input_path}")
            return
        
    except Exception as e:
        print(f"Failed for {input_path} with {str(e)}")
        raise e
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