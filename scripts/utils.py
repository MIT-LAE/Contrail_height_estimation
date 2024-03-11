"""
Helper functions that are used by the scripts within this directory.
"""
import os
import sys
import glob
import multiprocessing
import datetime as dt

import pandas as pd
import numpy as np

def load_dataframe(path):
    """
    Utility function to read a dataframe, without specifying the extension
    of the file. Used for 'backwards compatability' with previously
    generated pickle files.
    """
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unknown file type: {path}")

def process_multiple(process_single, input_path, save_dir, input_suffix,
                        output_suffix, parallel=False):
    """
    Process multiple files using `process_single`, either using serial
    or parallel processing. 

    Parameters
    ----------
    process_single : function
        The function that processes a single file.
    input_path : str
        The path to the input files.
    save_dir : str
        The directory to save the output files.
    input_suffix : str
        The suffix/extension of the input files.
    output_suffix : str
        The suffix of the output files.
    parallel : bool, optional
        Whether to use parallel processing.
    """
    # If input path is a directory, we'll look for all files with the
    # indicated extension
    if os.path.isdir(input_path):
        glob_path = os.path.join(input_path, "*" + input_suffix)
        paths = glob.glob(glob_path)
    # If the input path is a file and it has the desired input suffix,
    elif input_path.endswith(input_suffix):
        paths = [input_path]
    # Otherwise we assume the file contains a list of paths
    else:
        paths = pd.read_csv(input_path, header=None)[0].values
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create save names
    save_paths = []
    for p in paths:
        save_paths.append(os.path.join(save_dir,
                    os.path.basename(p).replace(input_suffix, output_suffix)))
        
    if not parallel:
        for p, s in zip(paths, save_paths):
            process_single(p, s)
    else:
        n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

        pool = multiprocessing.Pool(n_cpus)

        print(f"Running {__file__} in parallel using {n_cpus} CPUs")

        pool.starmap(process_single,
                        zip(paths, save_paths))
        pool.close()


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