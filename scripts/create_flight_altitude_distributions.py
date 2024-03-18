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
                                    subset_flight_data, get_haversine)

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
            # TODO: Remove ERA5 file directory and get this from
            # function instead
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

    # Create DomainBoundary object from the union
    db_flights = DomainBoundary(*np.column_stack(
                polygon_union.convex_hull.exterior.xy).T, great_circle=True)
    
    # Load OpenSky flight data
    flights = get_flight_data(time)

    # Rename columns
    flights = flights.rename(columns={"lat" : "latitude",
                                      "lon" : "longitude",
                                      "alt" : "altitude",
                                      "velocity" : "speed"})

    # Drop NaN values in specified columns
    flights = flights.dropna(subset=["longitude", "latitude", "time",
                                        "heading", "speed", "callsign"])
    
    # Perform subsetting to obtain flight candidates
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


def get_flight_altitude_distribution(flights):
    """
    For the flight data within `flights`, finds the distance flown in km
    on every 10th flight level (FL).

    Parameters
    ----------
    flights : pd.DataFrame
        Flight data
    
    Returns
    -------
    distance_distribution : pd.DataFrame
        Distance flown on every 10th flight level (FL), in km
    """

    def get_segment_length(x):
        
        # Obtain distances between waypoints in `x` in km
        dists = get_haversine(x["longitude"].values.astype(np.float64)[1:],
                        x["latitude"].values.astype(np.float64)[1:],
                        x["longitude"].shift(1).values.astype(np.float64)[1:],
                        x["latitude"].shift(1).values.astype(np.float64)[1:])
    
        # Add this information is a new column
        # Prepend a 0 to account for the first waypoint
        x["dist_km"] = np.hstack((0, dists))
        return x

    flights = flights.groupby("icao24",
                        group_keys=False).apply(get_segment_length)

    # / 0.3048 for conversion meters to feet
    # / 100 for conversion feet to flight level (FL)
    flights["FL"] = flights["altitude"] / 0.3048 / 100
    
    # Round flight levels to the nearest 10th
    flights["FL_bin"] = 10 * np.round(flights["FL"].values.astype(np.float32)\
                                      /10)

    distance_distribution = flights.groupby("FL_bin")[["dist_km"]].sum()

    return distance_distribution


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