#!/home/vmeijer/.conda/envs/gspy/bin/python -u

#SBATCH --time=1-00:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH -J flight_alt_dists
"""
Use this script to create flight altitude distributions.
"""
import os
import sys
import glob
import multiprocessing 
import datetime as dt
import tempfile

import click
import numpy as np
import pandas as pd
import shapely.affinity
from shapely.geometry import Point
from shapely.ops import unary_union

from contrails.meteorology.era5 import get_ERA5_data
from contrails.flights.utils import (DomainBoundary, get_flight_data,
                                    subset_flight_data)
from CAP.afca import (prepare_AFCA_input_flights,
                      prepare_AFCA_backward_analysis, SECONDS_IN_HOUR,
                      OODSENTINEL)
from utils import process_multiple

INPUT_SUFFIX = ".npy"
OUTPUT_SUFFIX = "_coarse_collocation.parquet"

AFCA_EXECUTABLE = "/home/vmeijer/AFCA/AFCA"

HOURS_BACKWARD = 2
MAX_WIND_ERROR = 10 # m/s
RADIUS_EARTH = 6371e3 # m

# Altitude limits for flight data
MIN_ALTITUDE = 8000 # m
MAX_ALTITUDE = 15e3 # m



def get_uncertainty_ellipse(longitude, latitude, time_advected,
                            dV=MAX_WIND_ERROR):  
    """
    Returns a shapely geometry representing the uncertainty ellipse
    for a given point on a trajectory, assuming the horizontal wind uncertainty
    of `dV`.
    
    Parameters
    ----------
    longitude : float
        Longitude of trajectory position, degrees
    latitude : float
        Latitude of trajectory position, degrees
    time_advected : float
        Time since start of trajectory, seconds
    dV : float (optional)
        Horizontal wind uncertainty, in m/s
    
    Returns
    -------
    uncertainty_ellipse : shapely.geometry.polygon.Polygon
        The ellipse representing the trajectory uncertainty
    """
    
    Δϕ = np.degrees(dV * time_advected / RADIUS_EARTH)
    Δλ = Δϕ / np.cos(np.radians(latitude))
    
    circle = Point(longitude, latitude).buffer(1)
    ellipse = shapely.affinity.scale(circle, Δλ, Δϕ)
    
    return ellipse


def get_flight_candidates(longitude, latitude, time,
                          hours_backward=HOURS_BACKWARD):
    """
    Obtains the flights that could possibly have formed a contrail found 
    at the given location and time

    Parameters
    ----------
    longitude : float
        Longitude of the contrail detection
    latitude : float
        Latitude of the contrail detection
    time : datetime
        Time of the contrail detection
    hours_backward : int, optional
        Number of hours to advect the contrail position backwards in time for

    Returns
    -------
    flights : pd.DataFrame
        Flight data
    """

    # The first step is to advect the contrail detection backwards in time
    # at all pressure levels found in the ERA5 data
    era5 = get_ERA5_data(time)
    pressure_levels = era5.isobaricInhPa.values

    polygons = []
    sub_polygons = []
    afca_outs = []

    n_press = len(pressure_levels)
    result = pd.DataFrame({"start_longitude" : [longitude] * n_press,
                        "start_latitude" : [latitude] * n_press,
                        "start_time" : [time] * n_press,
                        "start_pressure_hPa" : pressure_levels})
    with tempfile.TemporaryDirectory() as tmp_dir:
        AFCA_input_path = os.path.join(tmp_dir, "AFCA_input.csv")
        AFCA_output_path = os.path.join(tmp_dir, "AFCA_output.csv")

        for hour in range(1, hours_backward + 1):

            afca_inp = prepare_AFCA_backward_analysis([longitude], [latitude],
                                        np.array([time]),
                                        pressure_levels, n_hours=hour, step=1)
            
            afca_inp.to_csv(AFCA_input_path)
            # TODO: Remove ERA5 file directory and get this from function instead
            cmd = (f"{AFCA_EXECUTABLE} {AFCA_input_path} "
                f"/net/d15/data/vmeijer/ERA5/{time.strftime('%Y/%Y_%m_%d.nc')}"
                f" {AFCA_output_path} --ood-sentinel")

            exit_code = os.system(cmd)
            if exit_code != 0:
                raise RuntimeError(cmd)

            afca_out = pd.read_csv(AFCA_output_path)
            # Add results as new columns in `result' dataframe
            for colname in ["longitude", "latitude", "pressure_hPa"]:
                result[f"{colname}_{hour}"] = afca_out[colname]
    
    # Set out of domain data points to NaN values
    result[result == OODSENTINEL] = np.nan
    result["polygon_1"] = np.nan
    result["polygon_2"] = np.nan
    
    for i, row in result.iterrows():
        for hour in range(1, hours_backward + 1):
            suffix = f"_{hour}"
            lon = row["longitude" + suffix]
            lat = row["latitude" + suffix]

            # Skip if pressure level was out of domain.
            if np.any(np.isnan([lon, lat])):
                continue

            result.loc[i, "polygon" + suffix] = get_uncertainty_ellipse(lon,
                                                                        lat,
                                                        hour * SECONDS_IN_HOUR)

    # Drop pressure levels without polygons
    result = result.dropna(subset=['polygon_1', 'polygon_2'])

    # Take the union of the polygons
    polygon_union = unary_union(list(result.polygon_1.values) \
                                + list(result.polygon_2.values))

    db_flights = DomainBoundary(*np.column_stack(
                polygon_union.convex_hull.exterior.xy).T, great_circle=True)

    flights = get_flight_data(time)
    flights = flights.rename(columns={"lat" : "latitude",
                                      "lon" : "longitude",
                                      "alt" : "altitude",
                                      "velocity" : "speed"})

    flights = flights.dropna(subset=["longitude", "latitude", "time",
                                        "heading", "speed", "callsign"])
    flights = subset_flight_data(pd.DataFrame(flights),
                            boundary=db_flights,
                            min_altitude=MIN_ALTITUDE,
                            max_altitude=MAX_ALTITUDE,
                            min_time=time-dt.timedelta(hours=hours_backward),
                            max_time=time)
    return flights


def forward_advect_flights(flights, time):
    afca_inp = prepare_AFCA_input_flights(flights, time)

    with tempfile.TemporaryDirectory() as tmp_dir:
        AFCA_input_path = os.path.join(tmp_dir, "AFCA_input.csv")
        AFCA_output_path = os.path.join(tmp_dir, "AFCA_output.csv")

        afca_inp.to_csv(AFCA_input_path)
        # TODO: Remove ERA5 file directory and get this from function instead
        cmd = (f"{AFCA_EXECUTABLE} {AFCA_input_path} "
            f"/net/d15/data/vmeijer/ERA5/{time.strftime('%Y/%Y_%m_%d.nc')}"
            f" {AFCA_output_path} --ood-sentinel")

        exit_code = os.system(cmd)
        if exit_code != 0:
            raise RuntimeError(cmd)

        afca_out = pd.read_csv(AFCA_output_path)

    for k in ["longitude", "latitude", "pressure_hPa"]:

        flights["adv_" + k] = afca_out[k].values

    for col in ["latitude", "longitude", "altitude", "pressure_hPa", 
                "heading", "adv_pressure_hPa", "speed", 
                "delta_x", "delta_t", "v_dt"]:
        flights[col] = flights[col].values.astype(np.float64)

    return flights


    
def process_FL_dist(path):
    
    res = pd.read_pickle('test_set_glad-resonance.pkl')
    
    conus = False
    if "MCMIPC" in path:
        conus = True

    time = dt.datetime.strptime("_".join(os.path.basename(path).split("_")[1:4]), "%Y%m%d_%H_%M")
    min_col, max_col, min_row, max_row = [int(s) for s in os.path.basename(path).split(".")[0].split("_")[-4:]]
    label = np.load(path.replace("images", "labels"))

    rows, cols = np.where(label > 0)
    rows += min_row
    cols += min_col


    suffix = "C" if conus else "F"
    nc = xr.open_dataset(get_nc_path(time, product="ABI-L2-MCMIP" + suffix))

    x = nc.x.values[cols]
    y = nc.y.values[rows]

    lons, lats = ABI2geodetic(x, y)
    

    from collections import defaultdict

    itp_coords = {"longitude" : xr.DataArray(lons, dims="s"),
                  "latitude" : xr.DataArray(lats, dims="s"),
                  "time" : xr.DataArray(np.array([time] * len(lons)), dims="s")}

    era5 = get_ERA5_data(time)
    
    gh_vals = era5.z.interp(**itp_coords, method="linear").values

    pressure_df = defaultdict(list)

    keys = list(res.columns)
    for rm in ["error", "pe", "ape", "filename"]:
        keys.remove(rm)

    counter = 0
    for _, row in res[res.filename == path].iterrows():

        pressures = get_contrail_pressures(row[keys] * 1000, gh_vals[counter,:] / 9.80665, era5.isobaricInhPa.values)
        counter += 1
        for k, v in zip(keys, pressures):

            pressure_df[k].append(v)

    pressure_df = pd.DataFrame(pressure_df)
    FL_df = pres2alt(pressure_df * 100) / 0.3048 / 100
    FL_df.to_parquet("/home/vmeijer/contrail-height-estimation/data/FL_distributions/" \
                    + os.path.basename(path).replace(".png", ".parquet"),
                    )
    
    
    
def process_dist_distribution(path):
    
    
    save_path = "/home/vmeijer/contrail-height-estimation/data/flight_distributions/" \
        + os.path.basename(path).replace(".npy",".parquet")
    
    def dist_func(x):
    
        dist = np.hstack((0, get_haversine(x["longitude"].values.astype(np.float64)[1:],
                                           x["latitude"].values.astype(np.float64)[1:],
                                           x["longitude"].shift(1).values.astype(np.float64)[1:],
                                           x["latitude"].shift(1).values.astype(np.float64)[1:])))

        x["dist_km"] = dist
        return x
    
    filtered_flights = pd.read_parquet(save_path)
                     
    filtered_flights = filtered_flights.groupby("icao24", group_keys=False).apply(dist_func)

    filtered_flights["FL"] = filtered_flights["altitude"] / 0.3048 / 100
    filtered_flights["FL_bin"] = 10 * np.round(filtered_flights["FL"].values.astype(np.float32)/10)

    hist_dist = filtered_flights.groupby("FL_bin")[["dist_km"]].sum()
    hist_dist.to_parquet(save_path.replace("flight", "distance"))
    
    
def process_weather_data(path):
    
    conus = False
    if "MCMIPC" in path:
        conus = True

    time = dt.datetime.strptime("_".join(os.path.basename(path).split("_")[1:4]), "%Y%m%d_%H_%M")
    min_col, max_col, min_row, max_row = [int(s) for s in os.path.basename(path).split(".")[0].split("_")[-4:]]
    label = np.load(path.replace("images", "labels"))

    rows, cols = np.where(label > 0)
    rows += min_row
    cols += min_col


    suffix = "C" if conus else "F"
    nc = xr.open_dataset(get_nc_path(time, product="ABI-L2-MCMIP" + suffix))

    x = nc.x.values[cols]
    y = nc.y.values[rows]

    lons, lats = ABI2geodetic(x, y)
    
    itp_coords = {"longitude" : xr.DataArray(lons, dims="s"),
                  "latitude" : xr.DataArray(lats, dims="s"),
                  "time" : xr.DataArray(np.array([time] * len(lons)), dims="s")}

    era5 = get_ERA5_data(time)
    
    itp = era5.interp(**itp_coords, method="linear")
    
     
    save_path = "/home/vmeijer/contrail-height-estimation/data/weather_itp/" \
        + os.path.basename(path).replace(".png", ".nc")
    
    itp.to_netcdf(save_path)
    

def get_contrail_pressures(heights, gh_vals, pressures):
    contrail_pressures = np.zeros_like(heights)
    
    for i in range(len(heights)):
        
        contrail_pressures[i] = np.interp(heights[i], gh_vals, pressures)
        
        
    return contrail_pressures
    

    

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