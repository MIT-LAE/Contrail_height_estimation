#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH -J gen_figs
"""
Use this script to generate figures for the manual inspection of contrail
collocations in the CALIOP L1 data.
"""

import sys
import os
import glob

import click
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.cm as cm
import cartopy.crs as ccrs
from skimage.morphology import binary_dilation

from CAP.caliop import CALIOP
from CAP.visualization import plot_caliop_curtain
from CAP.iir import IIR
from CAP.abi import ORTHO_PROJ, ORTHO_EXTENT
from CAP.utils import get_ortho_ids, get_lons, get_lats

from contrails.satellites.goes.reprojection import get_reprojection_path
from contrails.satellites.goes.abi import ash_from_nc, ash_from_h5, get_image_path

from utils import process_multiple, load_dataframe, get_mask

# Set font to serif
plt.rc("font", family="serif")

IIR_DIR = "/net/d15/data/vmeijer/IIR_L1/"
L1_ROOT = "/net/d15/data/vmeijer/CALIOP_L1/"
ORTHO_IDS = get_ortho_ids()


INPUT_SUFFIX = "_fine_collocation.parquet"
OUTPUT_SUFFIX = ".png"

def generate_L1_IIR_figure(df, save_name=None):

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
    cloud_mask, b532, b1064, lons, lats, times = ca.get_cloud_mask(
                                                    return_backscatters=True,
                                                    ve1=8, ve2=15,
                                                    extent=extent)

    extent[0] = lons.min()
    extent[1] = lons.max()

    ascending = lats[-1] > lats[0]

    data = b532
    data = np.ma.masked_invalid(data)

    filtered = (data * cloud_mask).T

    fig = plt.figure(dpi=600, figsize=(7.2, 7.2))

    grid = GridSpec(1, 11)

    ax = fig.add_subplot(grid[0,:2])

    if ascending:
        plot_caliop_curtain(fig, ax, lons, lats, times, filtered.T,
                        min_alt=8, max_alt=15, rotate=True)
    # If descending, reverse the order of the data
    else:
        plot_caliop_curtain(fig, ax, lons[::-1],
                        lats[::-1], times[::-1], filtered.T[:,::-1],
                        min_alt=8, max_alt=15, rotate=True)

    # Modify horizontal axis tick labels
    ax.set_xticks([8.0, 11.5, 15.0])
    ax.minorticks_off()

    ax.set_aspect("auto")
    pos = ax.get_position()
    ax_height = 15 * (pos.y1 - pos.y0) * 300


    ax1 = fig.add_subplot(grid[0,4:6])

    iir = IIR(get_IIR_path(ca))

    iir_BT2, iir_lons, iir_lats = iir.get_BT_image(extent, channel=2)
    iir_BT3 = iir.get_BT_image(extent, channel=3, return_coords=False)
    BTD = iir_BT2 - iir_BT3

    middle_iir_lats = iir_lats[:,35]

    ax1.imshow(iir_BT2, cmap="gray")
    ax1.axvline(35, c="k")
    
    ax2 = fig.add_subplot(grid[0,6:8])
    ax2.imshow(BTD, cmap="gray")
    ax2.axvline(35, c="k")

    gax = fig.add_subplot(grid[0,8:], projection=ORTHO_PROJ)

    if "CONUS" in goes_product:
        suffix = "C"
        conus = True
    else:
        suffix = "F"
        conus = False

    if conus or goes_time > dt.datetime(2021, 12, 31):

        nc = xr.open_dataset(get_reprojection_path(goes_time, product="ABI-L2-MCMIP" + suffix))
        ash = ash_from_nc(nc)
    else:
        h5_path = get_image_path(goes_time)
        ash = ash_from_h5(h5_path)

    if len(ash.shape) != 3:
        ash = ash.reshape((2000, 3000, 3))

    mask = get_mask(goes_time,
                    conus=conus)

    boundaries = binary_dilation(mask) - mask

    gax.imshow(ash, extent=ORTHO_EXTENT, origin='upper', transform=ORTHO_PROJ)
    gax.imshow(np.ma.masked_array(boundaries, mask=(boundaries==0.)), extent=ORTHO_EXTENT,
               origin='upper', transform=ORTHO_PROJ, cmap="gray_r")

    gax.set_extent(extent, ccrs.PlateCarree())
    gax.set_aspect('auto')

    gl = gax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.left_labels = False 


    gax.plot(lons, lats, transform=ccrs.PlateCarree(), c="w", linewidth=2.5,
             label="CALIPSO ground track", zorder=2)
    gax.plot(lons, lats, transform=ccrs.PlateCarree(), c="k", linewidth=2,
             label="CALIPSO ground track", zorder=3) 

    # Plot matches, group them by unique GOES ids
    unique_ids = np.unique(df["goes_ABI_id"])

    if ascending:
        first_profile_id = np.where(ca.get("Latitude").data == lats[0])[0]
    else:
        first_profile_id = np.where(ca.get("Latitude").data == lats[-1])[0]

    counter = 0
    for i, ID in enumerate(unique_ids):
        r, c = get_ortho_row_col(ID)

        lo, la = goes_lons[r,c], goes_lats[r,c]
        clons = df[df.goes_ABI_id == ID].caliop_lon.values
        clats = df[df.goes_ABI_id == ID].caliop_lat.values

        # Map to IIR row
        if ascending:
            iir_rows = np.interp(clats, middle_iir_lats, np.arange(middle_iir_lats.size))
        else:
            iir_rows = np.interp(clats, middle_iir_lats[::-1], np.arange(middle_iir_lats.size)[::-1])

        c = cm.get_cmap("nipy_spectral")(i/len(unique_ids))

        gax.scatter(lo, la, c=[c], s=1, marker="s", transform=ccrs.PlateCarree(), zorder=4)
        gax.scatter(clons, clats, c=[c],s=1, marker="x", transform=ccrs.PlateCarree(), zorder=4)

        ax1.scatter([35] * len(iir_rows), iir_rows, s=1, c=[c], marker="s", zorder=4)
        ax2.scatter([35] * len(iir_rows), iir_rows, s=1, c=[c], marker="s", zorder=4)

        if ascending:
            ax.scatter(df[df.goes_ABI_id == ID].caliop_top_height.values/1000. +0.25,
                    (df[df.goes_ABI_id == ID].profile_id.values-first_profile_id),
                    c=[c], s=0.1)

        else:
            ax.scatter(df[df.goes_ABI_id == ID].caliop_top_height.values/1000. +0.25,
                    -(df[df.goes_ABI_id == ID].profile_id.values-first_profile_id),
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

    ax1.set(title=r"IIR 10.6 $\mu$m")
    if ascending:
        ax1.invert_yaxis()
    
    if iir_lons[0,-1] < iir_lons[0,0]:
        ax1.invert_xaxis()
    ax1.set_aspect("auto")
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    
    ax2.set(title=r"IIR BTD")
    if ascending:
        ax2.invert_yaxis()
    
    if iir_lons[0,-1] < iir_lons[0,0]:
        ax2.invert_xaxis()
    ax2.set_aspect("auto")
    ax2.set_xticks([])
    ax2.set_yticks([])

    if save_name is None:
        return fig
    else:
        plt.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close()
        del fig 

        
def get_IIR_path(ca):
    """
    Utility function to get IIR image path from caliop object.

    Parameters
    ----------
    ca : CALIOP
        CALIOP object for which to obtain path to corresponding IIR image
    
    Returns
    -------
    iir_path : str
        Path to IIR image
    """
    l1_path = ca.path

    if "2021" in l1_path or "2022" in l1_path:
        version = "V2-01"
    else:
        version = "V2-00"
    
    replace_pairs = [("CALIOP_L1", "IIR_L1"), ("V4-51", version),
                    ("V4-10", version), ("V4-11", version),
                    ("LID", "IIR"), ("ValStage1-V3-41", "Standard-V2-01"),
                    ("_Subset", "")]
    
    iir_filename = os.path.basename(l1_path)
    for old, new in replace_pairs:
        iir_filename = iir_filename.replace(old, new)

    iir_path = os.path.join(IIR_DIR, iir_filename)
    return iir_path
    

def get_ortho_row_col(ID):
    
    r, c = np.where(ORTHO_IDS==ID)

    return r[0], c[0]


def process_file(input_path, save_path):

    save_dir = os.path.dirname(save_path)
    
    try:
        df = load_dataframe(input_path)
        if len(df) == 0:
            return
        
        df["caliop_time"] = pd.to_datetime(df["caliop_time"])
        df["goes_time"] = pd.to_datetime(df["goes_time"])
        
        print(f"Started generating figure for {input_path}")
        df["segment_number"] = df.groupby("segment_start_lat").ngroup()
    
        for i, seg in enumerate(df.segment_number.unique()):
            
            save_name = os.path.join(save_dir,
                os.path.basename(input_path).replace(INPUT_SUFFIX, 
                f"_{i+1}" + OUTPUT_SUFFIX))
            
            fig = generate_L1_IIR_figure(df[df.segment_number == seg])
            fig.savefig(save_name, dpi=600, bbox_inches="tight")

        print(f"Finished generating figure for {input_path}")

    except Exception as e:
        print(f"Failed for {input_path} with {e}")
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