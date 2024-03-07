"""
Helper functions that are used by the scripts within this directory.
"""
import os
import sys
import glob
import multiprocessing

import pandas as pd

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
        glob_path = os.path.join(input_path, "*", input_suffix)
        paths = glob.glob(glob_path)
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
                    os.path.basename(p).replace(".hdf", output_suffix)))
        
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