#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH -J append_goes
from multiprocessing.sharedctypes import Value
import numpy as np, pandas as pd, datetime as dt, xarray as xr
import glob, os, multiprocessing

from contrails.satellites.goes.reprojection import *
from contrails.satellites.goes.abi import *

from pickle import UnpicklingError

FORCE_NEW = True


def main(p):
    save_path = p.replace("L1_manually_inspected", "final")
    
    if os.path.exists(save_path) and not FORCE_NEW:
        print(f"already processed {p}")
        return
    try:
        df = pd.read_csv(p)
        df = df[df.manual_inspection == "correct"]
    except UnpicklingError:
        print(f"Unpickling error for {p}")
        return
        
    #df['conus_time'] = df["time"].dt.round("5min")
    if len(df) == 0:
        return
    cols = []
    for i in range(1,17):
        lab = f"CMI_{str(i).rjust(2,'0')}"
        df[lab] = np.nan
        cols.append(lab)
    
    counter = 0
    for ctime in df.goes_time.unique():
        idx = df.goes_time == ctime
        try:
            nc = xr.open_dataset(get_nc_path(pd.Timestamp(ctime).to_pydatetime(), product="ABI-L2-MCMIPC"))
        except FileNotFoundError:
            continue
        except ValueError:
            print(ctime)
            print(p)
            raise ValueError
        
        x = xr.DataArray(df[idx].goes_ABI_col.values.astype(np.int64) - CONUS_FIRST_COL, dims="s")
        y = xr.DataArray(df[idx].goes_ABI_row.values.astype(np.int64) - CONUS_FIRST_ROW, dims="s")
        df.loc[idx,cols] = nc[[f"CMI_C{str(i).rjust(2,'0')}" for i in range(1,17)]].isel(x=x, y=y).to_array().values.T

    
    df.to_pickle(save_path)

if __name__ == "__main__":
    import sys, multiprocessing

    paths = np.sort(glob.glob("/home/vmeijer/height_estimation/data/L1_manually_inspected/2021*.csv"))

    if sys.argv[-1] == "DEBUG":
        for p in paths:
            main(p)

    else:
        pool = multiprocessing.Pool(os.cpu_count())
        pool.map(main, paths[::-1])
        pool.close()
    
