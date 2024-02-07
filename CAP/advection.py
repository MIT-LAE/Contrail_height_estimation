import numpy as np
import datetime as dt
from scipy.integrate import odeint

from .constants import RADIUS_EARTH
from .interpolation import interpolate_4d

def interpolate_winds(lon, lat, pressure, time,
        u, v, longitudes, latitudes,
        pressures_k, times):
    """
    Interpolate horizontal winds in a 4 dimensional wind field.

    For the horizontal interpolation, bicubic interpolation is used.
    For the vertical interpolation, quadratic interpolation is used.
    For the temporal interpolation, linear interpolation is used.

    Parameters
    ----------
    lon: float
        Longitude in degrees
    lat: float
        Latitude in degrees
    pressure: float
        Pressure in hPa
    time: dt.datetime
        Time in UTC
    u: np.array
        Eastward wind field (4 dimensions: time, pressure, lat, lon) in m/s
    v: np.array
        Northward wind field (4 dimensions: time, pressure, lat, lon) in m/s
    longitudes: np.array
        Longitudes in degrees associated with wind field, in ascending order
    latitudes: np.array
        Latitudes in degrees associated with wind field, in descending order
    pressures_k: np.array
        Pressures in hPa associated with wind field
    times: np.array
        Times in UTC associated with wind field
    
    Returns
    -------
    u_itp: float
        Interpolated eastward wind in m/s
    v_itp: float
        Interpolated northward wind in m/s
    """

    # Get indices of grid cell containing the point to interpolate
    # Calculation for 'i' assumes latitudes are descending
    i = np.where(lat - latitudes  > 0)[0][0] - 1
    j = np.where(lon - longitudes < 0)[0][0] - 1
    k = max(np.where(pressures_k - pressure < 0)[0][0] - 1,0)
    l = np.where((np.datetime64(time) - times) \
                     / np.timedelta64(1,'s') <= 0)[0][0] - 1
    
    # Subset the cell coordinates
    lons = longitudes[j-1:j+3]
    lats = latitudes[i-1:i+3]
    pressures = pressures_k[k:k+3]

    # Time difference in seconds
    Dt = np.float64((np.datetime64(time,'ns') - times[l]) \ 
                            / np.timedelta64(1,'s') /3600.)

    # Reverse order of latitudes for interpolation
    u_itp = interpolate_4d(lon, lat, pressure, Dt,
                            u[l:l+2, k:k+3, i-1:i+3, j-1:j+3],
                            lons, 
                            lats[::-1],
                            pressures,
                            np.arange(2.))

    v_itp = interpolate_4d(lon, lat, pressure, Dt,
                            v[l:l+2, k:k+3, i-1:i+3, j-1:j+3],
                            lons, 
                            lats[::-1],
                            pressures,
                            np.arange(2.))
    return u_itp, v_itp


def advection_rhs(x, t, t0, pressure, u, v, longitudes, latitudes,
                pressures_k, times):

    """
    Evaluates the RHS of the trajectory equation, in geodetic coordinates.
    Assumes isobaric advection, as the times over which advection is performed
    within this work are small (< 5 minutes).

    Parameters
    ----------
    x: np.array
        Position vector containing (lon, lat) in degrees
    t: float
        Time since starting trajectory analysis in seconds
    t0: dt.datetime
        Time trajectory analysis was started
    pressure: float
        Pressure in hPa
    u: np.array
        Eastward wind field (4 dimensions: time, pressure, lat, lon) in m/s
    v: np.array
        Northward wind field (4 dimensions: time, pressure, lat, lon) in m/s
    longitudes: np.array
        Longitudes in degrees associated with wind field
    longitudes: np.array
        Longitudes in degrees associated with wind field
    latitudes: np.array
        Latitudes in degrees associated with wind field
    pressures_k: np.array
        Pressures in hPa associated with wind field
    times: np.array
        Times in UTC associated with wind field
    Returns
    -------
    dx: np.array
        RHS of trajectory analysis equation
    """
    dx = np.zeros_like(x)

    nptime = np.datetime64(t0 + dt.timedelta(seconds=t), 'ns')

    try:
        u, v = interpolate_winds(x[0], x[1], pressure, nptime, u, v,
                            longitudes, latitudes, pressures_k, times)

        # Factor 180 / pi converts the resulting tendencies
        # to degrees per second
        rad2deg = 180 / np.pi
        dx[0] = rad2deg * (1 / (RADIUS_EARTH * np.cos(np.radians(x[1])))) * u
        dx[1] = rad2deg * (1 / RADIUS_EARTH) * v
    except IndexError:
        dx[0] = np.nan
        dx[1] = np.nan
    return dx

def get_advected_positions(lons, lats, times, heights, pressures, weather,
                            adv_time):
    """
    Get the advected positions of the CALIOP profiles at the time 'adv_time'
    using the ERA5 winds.

    Parameters
    ----------
    lons: np.array
        Longitude of CALIPSO ground track, degrees
    lats: np.array
        Latitude of CALIPSO ground track, degrees
    times: np.array
        Times of CALIPSO ground track, UTC
    heights: np.array
        Geometric altitude in meters
    pressures: np.array
        Pressure in hPa
    weather: xr.Dataset
        ERA5 weather dataset
    adv_time: dt.datetime
        Time to advect to
    
    Returns
    -------
    adv_lons: np.array
        Advected longitudes
    adv_lats: np.array
        Advected latitudes
    """
    
    # Unpack data from dataset and convert to float64
    # to comply with `advection_rhs` function.
    u = weather.u.values.astype(np.float64)
    v = weather.v.values.astype(np.float64)
    longitudes = weather.longitude.values.astype(np.float64)
    latitudes = weather.latitude.values.astype(np.float64)
    pressures_k = weather.isobaricInhPa.values.astype(np.float64)
    times_l = weather.time.values
    
    adv_lons = np.nan*np.zeros_like(pressures)
    adv_lats = np.nan*np.zeros_like(pressures)
    indices = np.where(~np.isnan(heights))[0]
    for i in indices:
        x0 = np.array([lons[i], lats[i]])
        Dt = [(adv_time - times[i]).total_seconds()]
        t_ode = np.hstack((np.array([0.0]), np.array(Dt)))
        sol = odeint(advection_rhs, x0, t_ode,
                    args=(times[i], pressures[i], u, v, longitudes, latitudes,
                    pressures_k, times_l))
        
        adv_lons[i] = sol[1,0]
        adv_lats[i] = sol[1,1]

    return adv_lons, adv_lats

        
def get_interpolated_winds(lons, lats, times, heights, pressures, weather):
    """
    Interpolate the ERA5 winds to the CALIOP profile locations.

    Parameters
    ----------
    lons: np.array
        Longitude of CALIPSO ground track, degrees
    lats: np.array
        Latitude of CALIPSO ground track, degrees
    times: np.array
        Times of CALIPSO ground track, UTC
    heights: np.array
        Geometric altitude in meters
    pressures: np.array
        Pressure in hPa
    weather: xr.Dataset
        ERA5 weather dataset

    Returns
    -------
    us: np.array
        Interpolated zonal winds
    vs: np.array
        Interpolated meridional winds
    """
    
    
    era5_lons = weather.longitude.values.astype(np.float64)
    era5_lats = weather.latitude.values.astype(np.float64)
    era5_pressures = weather.isobaricInhPa.values.astype(np.float64)
    era5_times = weather.time.values
    u = weather.u.values.astype(np.float64)
    v = weather.v.values.astype(np.float64)

    us = np.nan*np.zeros_like(pressures)
    vs = np.nan*np.zeros_like(pressures)
    indices = np.where(~np.isnan(heights))[0]
    for i in indices:
        u_itp, v_itp = interpolate_winds(lons[i], lats[i], pressures[i],
                                        times[i], u, v, era5_lons, era5_lats,
                                        era5_pressures, era5_times)

        us[i] = u_itp
        vs[i] = v_itp

    return us, vs