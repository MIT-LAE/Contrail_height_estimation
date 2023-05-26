#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH -J gen_figs

import sys, os, glob
import numpy as np, datetime as dt, pandas as pd, matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__) + "../src/")

from caliop import *
from collocation import *
from utils import *
from visualization import *

from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

import matplotlib.cm as cm
import cartopy.crs as ccrs


from skimage.morphology import binary_dilation
from contrails.satellites.goes.reprojection import get_reprojection_path

from contrails.satellites.goes.abi import ash_from_nc, ash_from_h5, get_image_path

L1_ROOT = "/net/d13/data/vmeijer/data/CALIPSO/CALIOP_L1/"
ORTHO_IDS = get_ortho_ids()

SAVE_DIR = "/home/vmeijer/contrail-height-estimation/figures/L1_collocation/"


def collocation_plot(df, save_name=None):

    
    goes_lons = get_lons()
    goes_lats = get_lats()
    
    if len(np.unique(df["segment_start_lat"])) > 1:
        raise ValueError("Can only analyze single CALIOP segments")
    
    L1_file = L1_ROOT + df.iloc[0].L1_file
    goes_time = df.iloc[0].goes_time
    goes_product = df.iloc[0]['goes_product']
    
    min_lat = min(df.iloc[0].segment_start_lat, df.iloc[0].segment_end_lat)
    max_lat = max(df.iloc[0].segment_start_lat, df.iloc[0].segment_end_lat)
    
    extent = [-135, -45, min_lat, max_lat]
    
    ca = CALIOP(L1_file)

        
    # Technically don't need to do this
    cloud_mask, b532, b1064, lons, lats, times = ca.get_cloud_filter(return_backscatters=True,
                                                                         min_alt=8, max_alt=15,
                                                                    extent=extent)
    
    extent[0] = lons.min()
    extent[1] = lons.max()
    
    ascending = ca.is_ascending()
    
    data = b532
    data = np.ma.masked_invalid(data)
    
    filtered = data.T*cloud_mask
    
    fig = plt.figure(dpi=600, figsize=(7.2, 7.2))
    
    grid = GridSpec(1, 6)

    ax = fig.add_subplot(grid[0,:2])

    if ascending:
        ca.plot_backscatter(fig=fig, ax=ax, extent=extent, min_alt=8, max_alt=15, rotate=True,
                           cloud_filter=True)
    else:
        ca.plot_backscatter(fig=fig, ax=ax, extent=extent, min_alt=8, max_alt=15, rotate=True,
                           reverse=True, cloud_filter=True)
    
    
    # Modify horizontal axis tick labels
    ax.set_xticks([8.0, 11.5, 15.0])
    ax.minorticks_off()

    ax.set_aspect("auto")
    pos = ax.get_position()
    ax_height = 15 * (pos.y1 - pos.y0) * 300
    
    gax = fig.add_subplot(grid[0,3:], projection=ORTHO_PROJ)
    
    if "CONUS" in goes_product:
        suffix = "C"
        conus = True
    else:
        suffix = "F"
        conus = False
        
    if conus:
        nc = xr.open_dataset(get_reprojection_path(goes_time, product="ABI-L2-MCMIP" + suffix))
        ash = ash_from_nc(nc)
    else:
        h5_path = get_image_path(goes_time)
        ash = ash_from_h5(h5_path)

    if len(ash.shape) != 3:
        ash = ash.reshape((2000, 3000, 3))
        
    mask = get_mask(goes_time, conus=conus)
    
    boundaries = binary_dilation(mask) - mask
    
    gax.imshow(ash, extent=ORTHO_EXTENT, origin='upper', transform=ORTHO_PROJ)
    gax.imshow(np.ma.masked_array(boundaries, mask=(boundaries==0.)), extent=ORTHO_EXTENT,
               origin='upper', transform=ORTHO_PROJ, cmap="gray_r")
    
    gax.set_extent(extent, ccrs.PlateCarree())
    gax.set_aspect('auto')
    
    gl = gax.gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False 

    
    gax.plot(lons, lats, transform=ccrs.PlateCarree(), c="w", linewidth=2.5,
             label="CALIPSO ground track", zorder=2)
    gax.plot(lons, lats, transform=ccrs.PlateCarree(), c="k", linewidth=2,
             label="CALIPSO ground track", zorder=3) 


    # Plot matches, group them by unique GOES ids
    unique_ids = np.unique(df["goes_ABI_id"])
    
    first_profile_id = np.where(ca.get("Latitude").data == lats[0])[0]
    
    counter = 0
    for i, ID in enumerate(unique_ids):
        r, c = get_ortho_row_col(ID)
            
        lo, la = goes_lons[r,c], goes_lats[r,c]
        clons = df[df.goes_ABI_id == ID].caliop_lon.values
        clats = df[df.goes_ABI_id == ID].caliop_lat.values
        
        c = cm.get_cmap("nipy_spectral")(i/len(unique_ids))

        gax.scatter(lo, la, c=[c], s=1, marker="s", transform=ccrs.PlateCarree(), zorder=4)
        gax.scatter(clons, clats, c=[c],s=1, marker="x", transform=ccrs.PlateCarree(), zorder=4)
        
        if ascending:
            ax.scatter(df[df.goes_ABI_id == ID].caliop_top_height.values/1000. +0.25,
                    (df[df.goes_ABI_id == ID].profile_id.values-first_profile_id),
                    c=[c], s=0.1)
        else:
            ax.scatter(df[df.goes_ABI_id == ID].caliop_top_height.values/1000. +0.25,
                    filtered.shape[1] -(df[df.goes_ABI_id == ID].profile_id.values-first_profile_id),
                    c=[c], s=0.1)
        
    gax.scatter([],[],label="Collocated, measured by CALIOP", marker="x",c="k")
    gax.scatter([],[], label="Collocated, as viewed by GOES-16", marker="s", c="k"
               )
    ax.set(title="CALIOP L1")
    gax.set(title="GOES-16 Ash")
    
    fig.suptitle(f"ABI-L2-MCMIP{suffix} time: {goes_time}\nAdvection time: {df.iloc[0].adv_time}",
                 fontsize=8)
    
    ## Add size bar
    ax.add_patch(patches.Rectangle([14.0, 0], height=100+2*33.3, width=1.0, facecolor="w"))
    ax.plot([14.3, 14.3], [50, 50+2*33.3], c="k")
    ax.text(14.3, 50+33.3, "20 km", ha="right", va="center", rotation=90, c="k",
            fontsize=6)
    
    if save_name is None:
        return fig
    else:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        plt.close()
        del fig 

    
ORTHO_IDS = get_ortho_ids()
def get_ortho_row_col(ID):
    
    r, c = np.where(ORTHO_IDS==ID)

    return r[0], c[0]


def get_mask(time, conus=False):
    
    if conus:
        path = "/net/d13/data/vmeijer/data/orthographic_detections_goes16/" \
            + "ABI-L2-MCMIPC" + time.strftime("/%Y/%j/%H/%Y%m%d_%H_%M.csv")
        df = pd.read_csv(path) 
        mask = np.zeros((2000, 3000))
        mask[df.row.values, df.col.values] = 1
        return mask
    else:
        df = pd.read_csv("/home/vmeijer/covid19/data/predictions_wo_sf/" + time.strftime('%Y%m%d.csv'))
        df.datetime = pd.to_datetime(df.datetime)
        df = df[df.datetime == time]
        mask = np.zeros((2000, 3000))
        mask[df.x.values, df.y.values] = 1
        return mask 
    

def main(input_path):
    
    try:
        df = pd.read_pickle(input_path)
        
        if len(df) == 0:
            return
        
        print(f"Started generating figure for {input_path}")
        df["segment_number"] = df.groupby("segment_start_lat").ngroup()
    
        for i, seg in enumerate(df.segment_number.unique()):
            
            save_name = SAVE_DIR + os.path.basename(input_path).replace(".pkl","") + f"_{i+1}.png"
            
            if not os.path.exists(save_name):
                fig = collocation_plot(df[df.segment_number == seg])
                fig.savefig(save_name, dpi=600, bbox_inches="tight")

        print(f"Finished generating figure for {input_path}")

    except Exception as e:
        print(f"Failed for {input_path}")
        return 


if __name__ == "__main__":

    files = np.sort(glob.glob("/home/vmeijer/contrail-height-estimation/data/fine/*.pkl"))

    from multiprocessing import Pool
    import sys

    if sys.argv[-1] == "DEBUG":
        for p in files:
            main(p)
    else:
        n_cpus = os.environ.get("SLURM_CPUS_PER_TASK", 1)

        pool = Pool(n_cpus)

        print(f"Running {__file__} in parallel using {n_cpus} CPUs")

        pool.map(main, files)
        pool.close()

