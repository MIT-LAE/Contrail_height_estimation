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
import xarray as xr

from traffic.data import opensky
import pytz

from contrails.satellites.goes.abi import get_nc_path
from contrails.meteorology.era5 import get_ERA5_data
from contrails.flights.utils import (DomainBoundary, get_flight_data,
                                    subset_flight_data, get_haversine)

from CAP.abi import ABI2geodetic
from CAP.afca import (prepare_AFCA_input_flights,
                      prepare_AFCA_backward_analysis, SECONDS_IN_HOUR,
                      OODSENTINEL)

INPUT_SUFFIX = ".npy"
OUTPUT_SUFFIX = "_coarse_collocation.parquet"

AFCA_EXECUTABLE = "/home/vmeijer/AFCA/AFCA"

HOURS_BACKWARD = 2
MAX_WIND_ERROR = 10 # m/s
RADIUS_EARTH = 6371e3 # m

# Altitude limits for flight data
MIN_ALTITUDE = 8000 # m
MAX_ALTITUDE = 15e3 # m


def interpolate_flights(df: pd.DataFrame, times: pd.Series, numerical_column,
                        categorical_columns) -> pd.DataFrame:
    """
    Interpolates flights contained within 'df' to the times specified by
    'times'.
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe holding flight data
    times: pd.Series
        Pandas series holding time data in UTC
    Returns
    -------
    interpolated: pd.DataFrame
        Dataframe holding interpolated flight data
    """

    df.time = pd.to_datetime(df.time)
    new_dfs = []

    # Loop through unique icao24s
    for icao24 in df.icao24.unique():

        sub_df = df[df.icao24 == icao24].reset_index().set_index('time')
        
        
        to_add = pd.DataFrame({col : np.nan for col in sub_df.columns},
                                 index=times[(times >= sub_df.index.min())
                                             *(times <= sub_df.index.max())])

        concat = pd.concat([sub_df, to_add]).sort_index()

        concat[numerical_column] = concat[numerical_column].interpolate(
                                        method='index')
        concat[categorical_columns] = concat[categorical_columns].fillna(
                                        method='ffill')
        

        new_dfs.append(concat.reset_index().rename(columns={"level_0":"time"}))

    interpolated = pd.concat(new_dfs)[["time"] + numerical_column \
                                        + categorical_columns]
    return interpolated



def filter_waypoints_based_on_distance(flights, contrail_lon, contrail_lat,
                                       contrail_time, dV=10):
    """
    Filters waypoints based on distance to contrail observation, after advection.
    
    Parameters
    ----------
    flights : pd.DataFrame
        Flight data, including columns for advected positions
    contrail_lon : float
        Longitude of contrail observation
    contrail_lat : float
        Latitude of contrail observation
    contrail_time : dt.datetime
        Time at which contrail is observed at specified location
    dV : float
        Assumed wind speed error, in m / s
    
    Returns
    -------
    close_flights : pd.DataFrame
        Flight data filtered based on distance to contrail observation
    """
    
    # Columns to interpolate
    num_columns = ["longitude", "latitude", "adv_longitude", "adv_latitude",
                   "pressure_hPa", "altitude", "adv_pressure_hPa"]
    
    # Columns to include
    cat_columns = ["icao24", "callsign"]
    
    # Interpolate flights to 1 min frequency to reduce the amount of distance
    # 'missed' by subsetting 
    itp_flights = interpolate_flights(flights,
                        pd.date_range(contrail_time - dt.timedelta(hours=2),
                        contrail_time,freq="1min"), num_columns, cat_columns)
    
    dists_to_contrail = get_haversine(itp_flights['adv_longitude'].values,
                                    itp_flights['adv_latitude'].values,
                                    contrail_lon, contrail_lat)

    # Time in seconds over which the contrail would need to advect 
    # at the assumed wind speed error to reach a particular point
    # * 1000 to convert from km to m
    time_to_contrail = dists_to_contrail * 1000 / dV

    advection_time = (contrail_time - itp_flights['time']).dt.total_seconds()

    mask = time_to_contrail <= advection_time
    
    return itp_flights[mask]
    

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
    try:
        flights = get_flight_data(time)
         # Rename columns
        flights = flights.rename(columns={"lat" : "latitude",
                                        "lon" : "longitude",
                                        "alt" : "altitude",
                                        "velocity" : "speed"})

        # Drop NaN values in specified columns
        flights = flights.dropna(subset=["longitude", "latitude", "time",
                                            "heading", "speed", "callsign"])
    except FileNotFoundError:

        # Try to query OpenSky database directly
        t1 = time - dt.timedelta(hours=hours_backward)
        t2 = time 
        try:
            tflights = opensky.history(start=t1.replace(tzinfo=pytz.utc),
                                        stop=t2.replace(tzinfo=pytz.utc),
                                        bounds=list(polygon_union.bounds),
                                        other_params=" AND baroaltitude>8000 ",
                                        nautical_units=False, cached=False)

            tflights = tflights.resample("5min").eval()
            flights = tflights.data
            flights['time'] = flights['timestamp']
            flights = flights.dropna(subset=["altitude", "longitude",
                                                "latitude", "time"])
            flights = flights[flights.altitude < 15500]
            flights = flights.sort_values(by=['callsign','time'])

            # Remove time zone to comply with subset_flight_data function
            flights["time"] = flights["time"].dt.tz_localize(None)
            
            # Rename columns
            flights = flights.rename(columns={"track" : "heading",
                                            "groundspeed" : "speed"})
            # Drop columns
            flights = flights.drop(columns=["timestamp", "squawk", "alert",
                                "onground", "spi", "hour", "track_unwrapped"])
        except Exception as e:
            print(t1, t2, polygon_union.bounds)
            raise e
    
    # Perform subsetting to obtain flight candidates
    try:
        flights = subset_flight_data(pd.DataFrame(flights),
                                boundary=db_flights,
                                min_altitude=MIN_ALTITUDE,
                                max_altitude=MAX_ALTITUDE,
                                min_time=time-dt.timedelta(hours=hours_backward),
                                max_time=time)
    # If no flights are left after subsetting
    except ValueError:
        return pd.DataFrame()
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
        if col in flights.columns:
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


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("save_dir", type=click.Path())
def main(input_path, save_dir):

    df = pd.read_parquet(input_path)

    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "flights"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "distributions"), exist_ok=True)

    # Loop through test files and create flight altitude distributions
    for file in df['file'].values:
        conus = False
        if "MCMIPC" in file:
            conus = True
        time = dt.datetime.strptime("_".join(
                    os.path.basename(file).split("_")[1:4]), "%Y%m%d_%H_%M")
        
        flight_save_path = os.path.join(save_dir, "flights",
                                f"{time.strftime('%Y%m%d_%H_%M')}.parquet")
        dist_save_path = os.path.join(save_dir, "distributions",
                                f"{time.strftime('%Y%m%d_%H_%M')}.parquet")
        
        if os.path.exists(flight_save_path) and os.path.exists(dist_save_path):
            print(f"Already processed {file}")
            print("Skipping...")
            continue
        min_col, _, min_row, _ = [int(s) 
                for s in os.path.basename(file).split(".")[0].split("_")[-4:]]
        label = np.load(file.replace("images", "labels"))

        rows, cols = np.where(label > 0)
        rows += min_row
        cols += min_col

        suffix = "C" if conus else "F"
        nc = xr.open_dataset(get_nc_path(time,
                                            product="ABI-L2-MCMIP" + suffix))
        try: 
            x = nc.x.values[cols]
            y = nc.y.values[rows]
        except:
            continue
        lons, lats = ABI2geodetic(x, y)

        # arbitrarily take first point
        longitude = lons[0]
        latitude = lats[0]
        try:
            flights = get_flight_candidates(longitude, latitude, time)
        except RuntimeError:
            print(f"AFCA failed for {time}")
            continue

        if len(flights) == 0:
            print(f"No flights found for {time}")
            continue
        try:
            flights = forward_advect_flights(flights, time)
        except RuntimeError:
            print(f"AFCA failed for {time}")
            continue       

        # Filter flights based on distance to contrail location
        flights = filter_waypoints_based_on_distance(flights, longitude,
                                                    latitude, time)
        
        flights.to_parquet(flight_save_path,
                            use_deprecated_int96_timestamps=True)
        if len(flights) == 0:
            print(f"No flights found for {time}")
            continue
        df = get_flight_altitude_distribution(flights)
        df.to_parquet(dist_save_path,
                                use_deprecated_int96_timestamps=True)

                            

if __name__ == "__main__":
    main()