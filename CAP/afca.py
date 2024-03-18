"""
Functions to work with AFCA (Advection for Flight and Contrail Analysis)
"""
import pytz
import numpy as np
import pandas as pd

from .utils import alt2pres

SECONDS_IN_HOUR = 3600

# AFCA assigns this value to coordinates of trajectories that go out of the
# domain of the weather dataset provided
OODSENTINEL = 9999.0

def prepare_AFCA_backward_analysis(lons, lats, times, pressure_levels,
                                   n_hours=4, step=10):
    """
    Prepares input file for AFCA to perform backward trajectory analysis.

    Parameters
    ----------
    lons : np.array
        Longitudes
    lats : np.array
        Latitudes
    pressure_levels : np.array
        Pressure levels, in hPa
    n_hours : int
        Number of hours to advect the trajectories
    step : int
        Take every `step` coordinates from the input data for the advection
        analysis
    
    Returns
    -------
    inp : pd.DataFrame
        Input file for AFCA, as a pandas dataframe
    """
    
    n_profiles = len(lons)
    n_levels = len(pressure_levels)
    points = n_profiles//step

    start_lons = np.zeros((points, n_levels))
    start_lats = np.zeros((points, n_levels))
    start_times = np.zeros((points, n_levels))
    adv_times = start_times.copy()
    start_pressures = np.zeros((points, n_levels))

    for idx in range(0, points):

        start_lons[idx,:] = lons[step*idx]
        start_lats[idx,:] = lats[step*idx]
        start_pressures[idx,:] = pressure_levels

        start_times[idx,:] = times[step*idx].replace(tzinfo=pytz.utc
                                                     ).timestamp()
        adv_times[idx,:] = start_times[idx,0] - SECONDS_IN_HOUR * n_hours
        
    
    inp = pd.DataFrame({"longitude" : start_lons.flatten(),
                        "latitude" : start_lats.flatten(),
                        "pressure_hPa" : start_pressures.flatten(),
                        "t0" : start_times.flatten(), 
                        "tf" : adv_times.flatten(),
                        "interval": SECONDS_IN_HOUR \
                            * np.ones(start_lons.size)})
    return inp


def prepare_AFCA_input_flights(flights, advection_time,
                               additional_columns=None):
    """
    Prepares input file for AFCA to perform forward trajectory analysis.

    Parameters
    ----------
    flights : pd.DataFrame
        Flight data
    advection_time : int
        Time to advect the trajectories, in seconds
    additional_columns : list
        Additional columns to include in the input file
    
    Returns
    -------
    inp : pd.DataFrame
        Input file for AFCA, as a pandas dataframe
    """
    if 'pressure_hPa' not in flights.columns:
        # Divide by 100 to convert from Pa to hPa
        flights['pressure_hPa'] = alt2pres(flights["altitude"].values) / 100
    
    convert = lambda x : pd.Timestamp(x).to_pydatetime().replace(
                            tzinfo=pytz.utc).timestamp()

    start_times = np.array([convert(t) for t in flights.time.values])
    inp = pd.DataFrame({"longitude": flights.longitude.values,
                        "latitude": flights.latitude.values,
                        "pressure_hPa": flights.pressure_hPa.values, 
                        "t0": start_times,
                        "tf": convert(advection_time)\
                            * np.ones(len(flights), dtype=np.int64)})

    if additional_columns is not None:
        inp[additional_columns] = flights[additional_columns].values
    
    inp["advected_time_goal"] = advection_time
    return inp