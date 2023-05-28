#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#SBATCH -J append_aux

import numpy as np, pandas as pd, datetime as dt, xarray as xr
import glob, os, multiprocessing, sys

sys.path.append("/home/vmeijer/contrail-height-estimation/src/")

from caliop import *
from geometry import * 
from collocation  import *
from abi import *
from utils import *

import numpy as np, xarray as xr, pandas as pd, datetime as dt

from contrails.satellites.goes.abi import get_look_angles


SAVE_DIR = "/home/vmeijer/contrail-height-estimation/data/L2_ABI_aux/"

def append_surface_temperature_data(df):
    
    
    time = pd.Timestamp(df.iloc[0].caliop_time).to_pydatetime()
    
    ds = xr.open_dataset("/home/vmeijer/contrail-properties/data/skin_temperatures/ERA5_" \
                         + time.strftime("%Y%m.grib"), engine="cfgrib")
    
    itp_data = {"longitude" : xr.DataArray(df.caliop_lon.values, dims="s"),
                "latitude" : xr.DataArray(df.caliop_lat.values, dims="s"),
                "time" : xr.DataArray(df.caliop_time.values, dims="s")}
    
    df["T_surf"] = ds.skt.interp(**itp_data, method='linear').values
    
    return df

def append_land_sea_mask_data(df):

    ds = xr.open_dataset("/home/vmeijer/contrail-height-estimation/data/lsm.nc")
    
    itp_data = {"longitude" : xr.DataArray(df.caliop_lon.values, dims="s"),
                "latitude" : xr.DataArray(df.caliop_lat.values, dims="s"),
    }
    
    df["land_mask"] = ds.lsm.interp(**itp_data, method='linear').isel(time=0).values

    return df


def append_auxiliary_data(df):
    
    
    df["VZA"], _ = get_look_angles(df.caliop_lon.values, df.caliop_lat.values, 0)
    df["VZA"] = 90 - df["VZA"]
    
    df["cos_doy"] = np.cos(2*np.pi * df.caliop_time.dt.day_of_year/365)
    df["sin_doy"] = np.sin(2*np.pi * df.caliop_time.dt.day_of_year/365)
    
    return df



def main(input_path, save_path):

    if os.path.exists(save_path):
        print(f"Already done for {input_path}, result at {save_path}")
        return
    try:
        print(f"Started appending auxiliary data to {input_path}")
        df = append_surface_temperature_data(pd.read_pickle(input_path))
        df = append_auxiliary_data(df)
        df = append_land_sea_mask_data(df)
        
        if len(df) > 0:
        
            print(f"Finished appending auxiliary data to {input_path}")
            df.to_pickle(save_path)

        else:
            print(f"No valid collocations found for {input_path}")

    except Exception as e:
        print(f"Failed for {input_path} with {str(e)}")
        return

if __name__ == "__main__":

    paths = np.sort(glob.glob("/home/vmeijer/contrail-height-estimation/data/L2_ABI/*.pkl"))
    save_paths = np.array([SAVE_DIR + os.path.basename(p) for p in paths])

    if sys.argv[-1] == "DEBUG":
        for p, s in zip(paths, save_paths):
            main(p, s)
    else:
        n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

        pool = multiprocessing.Pool(n_cpus)

        print(f"Running {__file__} in parallel using {n_cpus} CPUs")

        pool.starmap(main, zip(paths, save_paths))
        pool.close()