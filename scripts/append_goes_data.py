#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH -J append_goes

import numpy as np, pandas as pd, datetime as dt, xarray as xr
import glob, os, multiprocessing, sys

sys.path.append("/home/vmeijer/contrail-height-estimation/src/")

from caliop import *
from geometry import * 
from collocation  import *
from abi import *
from utils import *

from contrails.satellites.goes.abi import get_nc_path

SAVE_DIR = "/home/vmeijer/contrail-height-estimation/data/L2_ABI/"


def append_ABI_data(df):
    """
    Append GOES-16 ABI infrared channel data to collocation dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Collocation dataframe, should have columns ['goes_product', 'goes_time']
    
    Returns
    -------
    df : pd.DataFrame
        Collocation dataframe with additional columns containing ABI infrared data
    """
    
    df["group_n"] = df.groupby(["goes_product", "goes_time"]).ngroup()
    
    cols = []
    for i in range(7,17):
        lab = f"CMI_C{str(i).rjust(2,'0')}"
        df[lab] = np.nan
        cols.append(lab)
        
    
    for g in df.group_n.unique():
        
        idx = df.group_n == g
        sub = df[idx]
        
        if sub.iloc[0].goes_product == "FD":
            product = "ABI-L2-MCMIPF"
        else:
            product = "ABI-L2-MCMIPC"
        
        goes_time = pd.Timestamp(sub.iloc[0].goes_time).to_pydatetime()
        
        nc = xr.open_dataset(get_nc_path(goes_time, product=product))
        
        
        if product.endswith("F"):
            x = xr.DataArray(df[idx].goes_ABI_col.values.astype(np.int64), dims="s")
            y = xr.DataArray(df[idx].goes_ABI_row.values.astype(np.int64), dims="s")
            
            df.loc[idx,cols] = nc[[f"CMI_C{str(i).rjust(2,'0')}" for i in range(7,17)]].isel(x=x, y=y).to_array().values.T
            
        else:
            x = xr.DataArray(df[idx].goes_ABI_col.values.astype(np.int64) - CONUS_FIRST_COL, dims="s")
            y = xr.DataArray(df[idx].goes_ABI_row.values.astype(np.int64) - CONUS_FIRST_ROW, dims="s")
            
            mask = (x.values < 0)+(x.values > 2499)+(y.values < 0)+(y.values > 1499)
            
            x = xr.DataArray(x.values[~mask], dims='s')
            y = xr.DataArray(y.values[~mask], dims='s')
            
            df.loc[np.where(idx)[0][~mask],cols] = nc[[f"CMI_C{str(i).rjust(2,'0')}" for i in range(7,17)]].isel(x=x, y=y).to_array().values.T
        
        
    return df.dropna(subset=cols)


def append_regional_max_ABI_data(df, region_size=29):
    """
    Append regional maxima for GOES-16 ABI infrared channel data to collocation dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Collocation dataframe, should have columns ['goes_product', 'goes_time']
    region_size : int (optional)
        The size of the region in which to take the maximum, in pixels.
    
    Returns
    -------
    df : pd.DataFrame
        Collocation dataframe with additional columns containing ABI infrared data
    """
    
    
    df["group_n"] = df.groupby(["goes_product", "goes_time"]).ngroup()

    df_cols = []
    for i in range(7,17):
        lab = f"CMI_C{str(i).rjust(2,'0')}_max"
        df[lab] = np.nan
        df_cols.append(lab)
    

    for g in df.group_n.unique():

        idx = df.group_n == g
        sub = df[idx]

        if sub.iloc[0].goes_product == "FD":
            product = "ABI-L2-MCMIPF"
        else:
            product = "ABI-L2-MCMIPC"

        goes_time = pd.Timestamp(sub.iloc[0].goes_time).to_pydatetime()

        nc = xr.open_dataset(get_nc_path(goes_time, product=product))


        if product.endswith("F"):

            cols = df[idx].goes_ABI_col.values.astype(np.int64)
            rows = df[idx].goes_ABI_row.values.astype(np.int64)

            reg_rows, reg_cols = get_image_tile_indices(rows, cols, region_size, (5424, 5424))

            shape = reg_rows.shape
            x = xr.DataArray(reg_cols.flatten(), dims="s")
            y = xr.DataArray(reg_rows.flatten(), dims="s")
            df.loc[idx,df_cols] = nc[[c.strip("_max") for c in df_cols]].isel(x=x, y=y).to_array().values.reshape(shape + (10,)).max(axis=0)



        else:

            cols = df[idx].goes_ABI_col.values.astype(np.int64) - CONUS_FIRST_COL
            rows = df[idx].goes_ABI_row.values.astype(np.int64) - CONUS_FIRST_ROW


            reg_rows, reg_cols = get_image_tile_indices(rows, cols, region_size, (1500, 2500))

            shape = reg_rows.shape
            x = xr.DataArray(reg_cols.flatten(), dims="s")
            y = xr.DataArray(reg_rows.flatten(), dims="s")
            df.loc[idx,df_cols] = nc[[c.strip("_max") for c in df_cols]].isel(x=x, y=y).to_array().values.reshape(shape + (10,)).max(axis=0)
    
    return df.dropna(subset=df_cols)




def main(input_path, save_path):

    if os.path.exists(save_path):
        print(f"Already done for {input_path}, result at {save_path}")
        return
    try:
        print(f"Started appending ABI data to {input_path}")
        df = append_ABI_data(pd.read_pickle(input_path))

        if len(df) > 0:
            df = append_regional_max_ABI_data(df)
        
            print(f"Finished appending ABI data to {input_path}")
            df.to_pickle(save_path)

        else:
            print(f"No valid collocations found for {input_path}")

    except Exception as e:
        print(f"Failed for {input_path} with {str(e)}")
        return

if __name__ == "__main__":

    paths = np.sort(glob.glob("/home/vmeijer/contrail-height-estimation/data/L2/*.pkl"))[:10]
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
