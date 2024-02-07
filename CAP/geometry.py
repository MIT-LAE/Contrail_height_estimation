import os

import numpy as np
from scipy.optimize import bisect, newton

from .abi import GOES16_params
from .constants import GRS80_PARAMS, RADIUS_EARTH

class GroundTrack:
    """
    Represents a satellite ground track using great circle paths.

    Parameters
    ----------
    lons : np.array
        Longitudes, degrees
    lats : np.arary
        Latitudes, degrees
    n_segments : int (optional)
        The number of great circle paths to use to represent
        the ground track
    """

    def __init__(self, lons, lats, n_segments=5):

        self.lons = lons
        self.lats = lats
        if lats[0] > lats[-1]:
            self.ascending = False
        else:
            self.ascending = True

        self.n_segments = n_segments
        self.setup_segments()
    
    def setup_segments(self):
        """
        Break up the ground track in great circle paths
        """
        
        lat_per_segment = (self.lats[-1] - self.lats[0]) / self.n_segments

        # First segment
        approx_lat = lat_per_segment + self.lats[0]
        
        # Find index
        idx = np.argmin(np.abs(self.lats - approx_lat))

        self.segments = [GreatCirclePath(self.lons[0], self.lats[0],
                            self.lons[idx], self.lats[idx])]

        for _ in range(1, self.n_segments-1):

            # Start position of new segment
            lon0 = self.lons[idx]
            lat0 = self.lats[idx]

            # Find new idx
            approx_lat = lat0 + lat_per_segment
            
            # Find index of end position of segment
            idx = np.argmin(np.abs(self.lats-approx_lat))
            self.segments.append(GreatCirclePath(lon0, lat0, 
                                    self.lons[idx], self.lats[idx]))
        
        # Final segment
        self.segments.append(GreatCirclePath(self.lons[idx], self.lats[idx],
                                             self.lons[-1], self.lats[-1]))

    
    def get_coordinates(self, n_points=100):
        """
        Return coordinates of ground track. 

        Parameters
        ----------
        n_points : int (optional)
            Number of coordinate pairs to return

        Returns
        -------
        lons : np.array
            Longitudes, degrees
        lats : np.array
            Latitudes, degrees
        """

        # Divide n_points over segments
        points_per_segment = int(n_points/self.n_segments)
   
        points = []

        for i in range(self.n_segments):
            coords = self.segments[i].get_coordinates(
                                    n_points=points_per_segment)
            points.append(coords)

        stacked = np.hstack(points)
        return stacked[0,:], stacked[1,:]

    def get_crosstrack_distance(self, lon, lat):
        """
        Computes the crosstrack distance to the given coordinates.

        Parameters
        ----------
        lon : Union[float, np.array]
            Longitude, degrees
        lat : Union[float, np.arrray]
            Latitude, degrees

        Returns
        -------
        dists : Union[float, np.array]
            Crosstrack distance in km
        """

        if isinstance(lon, float):

            # Find relevant segment
            for seg in self.segments:
                
                if (seg.lat0 <= lat)*(seg.lat1 >= lat) and self.ascending:
                    break
                if (seg.lat0 >= lat)*(seg.lat1 <= lat) and not self.ascending:
                    break

            return seg.get_crosstrack_distance(lon, lat)
        
        else:
            # Find relevant segment
            dists = np.nan * np.zeros_like(lat)
            for seg in self.segments:
                if self.ascending:
                    idx = (seg.lat0 <= lat)*(seg.lat1 >= lat)
                else:
                    idx = (seg.lat0 >= lat)*(seg.lat1 <= lat)
                    
                dists[idx] = seg.get_crosstrack_distance(lon[idx], lat[idx])
            
            return dists


class GreatCirclePath:
    """
    Represents a great circle path between two points. 

    Parameters
    ----------
    lon0: float
        Longitude of first point, degrees
    lat0: float
        Latitude of first point, degrees
    lon1: float
        Longitude of second point, degrees
    lat1: float
        Latitude of second point, degrees
    """
    
    def __init__(self, lon0, lat0, lon1, lat1):
        
        self.lon0 = lon0
        self.lat0 = lat0
        self.lon1 = lon1
        self.lat1 = lat1
    
    def get_crosstrack_distance(self, lon, lat):
        """
        Compute distance from points specified by (lon, lat) to great circle.

        Parameters
        -----------
        lon : Union[float, np.array]
            Longitude, degrees
        lat : Union[float, np.array]
            Latitude, degrees

        Returns
        -------
        dist : Union[float, np.array]
            Distance in km 
        """ 
        
        bearing_start_end = np.radians(get_bearing(self.lon0, self.lat0,
                                                    self.lon1, self.lat1))
        bearing_start_third = np.radians(get_bearing(self.lon0, self.lat0,
                                                        lon, lat))
        
        angular_distance = get_angular_distance(self.lon0, self.lat0, lon, lat)
        
        dist = np.arcsin(np.sin(angular_distance) \ 
                * np.sin(bearing_start_third-bearing_start_end)) * RADIUS_EARTH
        return dist
    
    def get_coordinates(self, n_points=100):
        """
        Sample coordinates along great circle path

        Parameters
        ----------
        n_points : int (optional)
            Number of coordinate pairs to return

        Returns
        -------
        lons : np.array
            Longitudes, degrees
        lats : np.array
            Latitudes, degrees
        """
        t = np.linspace(0, 1, n_points)
        return great_circle_intermediate_point(self.lon0, self.lat0,
                                                    self.lon1, self.lat1, t)
        

def ECEF2geodetic(x):
    """
    Converts a vector in the Earth Centered Earth Fixed (ECEF) coordinate frame
    to spherical coordinates

    Parameters
    ----------
    x: np.array
        ECEF coordinate vector
    
    Returns
    -------
    lon: float
        Longitude in degrees
    lat: float
        Latitude in degrees
    """
    lat = np.degrees(np.arcsin(x[2,:]))
    lon = np.degrees(np.arctan2(x[1,:], x[0,:]))
    return lon, lat


def get_bearing(lon0, lat0, lon1, lat1): 
    """
    Computes bearing of line extending from (lon0, lat0) to (lon1, lat1).
    
    Parameters
    ----------
    lon0: float
        Longitude of first point, degrees
    lat0: float
        Latitude of first point, degrees
    lon1: float
        Longitude of second point, degrees
    lat1: float
        Latitude of second point, degrees
    
    Returns
    -------
    bearing: float
        Bearing in degrees
    """ 
    a1 = np.sin(np.radians(lon1-lon0))*np.cos(np.radians(lat1))
    a2 = np.cos(np.radians(lat0))*np.sin(np.radians(lat1)) \
            - np.sin(np.radians(lat0))*np.cos(np.radians(lat1))\
            * np.cos(np.radians(lon1-lon0))
    
    return np.degrees(np.arctan2(a1, a2))


def get_haversine(lon0, lat0, lon1, lat1):
    """
    Computes length of great circle path extending from (lon0, lat0) to
    (lon1, lat1).
    
    Parameters
    ----------
    lon0: float
        Longitude of first point, degrees
    lat0: float
        Latitude of first point, degrees
    lon1: float
        Longitude of second point, degrees
    lat1: float
        Latitude of second point, degrees
    
    Returns
    -------
    haversine: float
        Haversine distance in km
    """ 
    return RADIUS_EARTH * get_angular_distance(lon0, lat0, lon1, lat1)
   
        
def get_angular_distance(lon0, lat0, lon1, lat1):
    """
    Computes central angle between great circle path extending from
    (lon0, lat0) to (lon1, lat1).
    
    Parameters
    ----------
    lon0 : float
        Longitude of first point, degrees
    lat0 : float
        Latitude of first point, degrees
    lon1 : float
        Longitude of second point, degrees
    lat1 : float
        Latitude of second point, degrees
    
    Returns
    -------
    angular distance : float
        Central angle in radians 
    """ 
    a = np.sin(0.5*np.radians(lat1-lat0))**2 \
        + np.cos(np.radians(lat0))*np.cos(np.radians(lat1)) \
            * np.sin(0.5*np.radians(lon1-lon0))**2

    angular_distance = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return angular_distance


def parallax_correction_vicente_forward(lon, lat, h,
                        globe_params=GRS80_PARAMS, sat_params=GOES16_PARAMS):
    """
    Implements the parallax correction method by Vicente et al. (2002) as
    described in Bielinski (2020). 

    Parameters
    ----------
    lon : float
        Longitude, degrees
    lat : float
        Latitude, degrees
    h : float
        Cloud height, meters
    globe_params : dict (optional)
        The ellipsoid used for geodetic coordinates
    sat_params : dict (optional)
        The satellite used for the correction.

    Returns
    -------
    lon_correct : float
        Corrected longitude, degrees
    lat_correct : float
        Corrected latitude, degrees
    """
    # Unpack parameters
    R_eq = globe_params["a"]
    R_pol = globe_params["b"]
    
    e = np.sqrt(R_eq**2-R_pol**2)/R_eq
    
    h_s = sat_params["h"]
    sat_lon = sat_params["lon_0"]
    
    
    R_equivalent = R_eq/np.sqrt(np.cos(np.radians(lat))**2 \
                            + (R_eq**2/R_pol**2)*np.sin(np.radians(lat))**2)
    
    # geocentric latitude
    glat = np.degrees(np.arctan((1-e**2)*np.tan(np.radians(lat))))
    
    # Apparent cloud position
    Xc = R_equivalent*np.cos(np.radians(glat))*np.cos(np.radians(lon-sat_lon))
    Yc = R_equivalent*np.cos(np.radians(glat))*np.sin(np.radians(lon-sat_lon))
    Zc = R_equivalent*np.sin(np.radians(glat))

    # Satellite position
    Xs = R_eq + h_s
    Ys = 0
    Zs = 0

    # Difference vector
    Xcs = Xs - Xc
    Ycs = Ys - Yc
    Zcs = Zs - Zc

   
    # Solve equation
    if isinstance(lat, float):
        # Determine equation to be solved
        f = lambda c: (((Xc+c*Xcs)**2 + (Yc+c*Ycs)**2)/(R_eq + h)**2) \
                + ((Zc + c*Zcs)**2/(R_pol +h)**2) - 1
        c = bisect(f, 0, 1)
    else:
        c = np.zeros_like(lat)
        for i in range(c.size):
            # Determine equation to be solved
            f = lambda c: (((Xc[i]+c*Xcs[i])**2 \
                + (Yc[i]+c*Ycs[i])**2)/(R_eq + h[i])**2) \
                + ((Zc[i] + c*Zcs[i])**2/(R_pol +h[i])**2) - 1
            try:
                c[i] = bisect(f, 0, 1)
            except ValueError:
                print(f(0), f(1))
                raise ValueError
            

    # Cloud top coordinates
    Xt = Xc + c*Xcs
    Yt = Yc + c*Ycs
    Zt = Zc + c*Zcs

    # Convert to geodetic
    lat_correct = np.degrees(np.arctan(((R_eq)**2/(R_pol)**2) \
                        * Zt/(np.sqrt(Xt**2 + Yt**2)) ) )
    lon_correct = np.degrees(np.arctan(Yt/Xt)) + sat_lon
    
    return lon_correct, lat_correct


def parallax_correction_vicente_backward(lon, lat, h, 
                        globe_params=GRS80_PARAMS, sat_params=GOES16_PARAMS):
    """
    Implements the inverse of the parallax correction method by Vicente et al.
    (2002) as described in Bielinski (2020). 

    That is, given a geodetic position and height, where would a given
    satellite  'view' the object?

    Parameters
    ----------
    lon : float
        Longitude, degrees
    lat : float
        Latitude, degrees
    h : float
        Cloud height, meters
    globe_params : dict (optional)
        The ellipsoid used for geodetic coordinates
    sat_params : dict (optional)
        The satellite used for the correction.

    Returns
    -------
    lon_correct : float
        Corrected longitude, degrees
    lat_correct : float
        Corrected latitude, degrees
    """
    
    # Unpack parameters
    R_eq = globe_params["a"]
    R_pol = globe_params["b"]
    
    h_s = sat_params["h"]
    sat_lon = sat_params["lon_0"]
    
    e = np.sqrt(R_eq**2-R_pol**2)/R_eq
    
    R_equivalent = (R_eq+h)/np.sqrt(np.cos(np.radians(lat))**2 \
                    + ((R_eq+h)**2/(R_pol+h)**2)*np.sin(np.radians(lat))**2)
    
    glat = np.degrees(np.arctan((1-e**2)*np.tan(np.radians(lat))))
    # Cloud position
    Xt = R_equivalent*np.cos(np.radians(glat))*np.cos(np.radians(lon-sat_lon))
    Yt = R_equivalent*np.cos(np.radians(glat))*np.sin(np.radians(lon-sat_lon))
    Zt = R_equivalent*np.sin(np.radians(glat))

    # Satellite position
    Xs = R_eq + h_s
    Ys = 0
    Zs = 0

    # Difference vector
    Xts = Xt - Xs
    Yts = Yt - Ys
    Zts = Zt - Zs

    # Solve equation
    if isinstance(lat, float) or isinstance(lat,int):
        # Determine equation to be solved
        f = lambda c: ((Xt+c*Xts)**2 + (Yt+c*Yts)**2)/(R_eq**2) \
                                + ((Zt + c*Zts)**2/(R_pol**2)) - 1
        f_prime = lambda c: 2*(Xts*(Xt+c*Xts)+Yts*(Yt+c*Yts))/R_eq**2 \
                        + 2*Zts*(Zt+c*Zts)/R_pol**2 
        
        c = newton(f, -1, fprime=f_prime)
    else:
        c = np.zeros_like(lat)
        for i in range(c.size):

            # Determine equation to be solved                                    
            f = lambda c: ( ((Xt[i]+c*Xts[i])**2 \
                            + (Yt[i]+c*Yts[i])**2)/(R_eq**2)) \
                            + ((Zt[i] + c*Zts[i])**2/(R_pol**2)) - 1
            f_prime = lambda c: 2*(Xts[i]*(Xt[i]+c*Xts[i])\
                + Yts[i]*(Yt[i]+c*Yts[i]))/R_eq**2 \
                    + 2*Zts[i]*(Zt[i]+c*Zts[i])/R_pol**2 
            try:
                c[i] = newton(f, -1, fprime=f_prime)
            except (ValueError, RuntimeError):
                print("Backward parallax correction failed")
                print(f"longitude: {lon[i]}, latitude: {lat[i]}," \
                        + f" height: {h[i]}")
                print(f"F(-1): {f(-1)}, F(0): {f(0)}")
                raise ValueError
                

    # Apparent cloud top coordinates
    Xc = Xt + c*Xts
    Yc = Yt + c*Yts
    Zc = Zt + c*Zts

    # Convert to geodetic
    lat_correct = np.degrees(np.arctan( (R_eq**2/(R_pol**2) \
                                * Zt/(np.sqrt(Xc**2 + Yc**2)) )))
    lon_correct = np.degrees(np.arctan(Yc/Xc)) + sat_lon
    
    return lon_correct, lat_correct


def geodesic_distance(lon1, lat1, lon2, lat2, globe_params=GRS80_PARAMS,
                         max_iter=100, tol=1e-6):
    """
    Uses Vicenty's formula to compute the geodesic distance between two
    geodetic points. This is more accurate than the Haversine distance, which
    assumes a spherical earth.

    Based on:
    https://github.com/maurycyp/vincenty/blob/master/vincenty/__init__.py

    For more information on the formulae, refer to the original publication by
    Vicenty:
    https://www.tandfonline.com/doi/abs/10.1179/sre.1975.23.176.88

    Parameters
    ----------
    lon1: float
        Longitude of first point, degrees
    lat1: float
        Latitude of first point, degrees
    lon2: float
        Longitude of second point, degrees
    lat2: float
        Latitude of second point, degrees
    globe_params: dict (optional)
        The ellipsoid used for geodetic coordinates
    max_iter: int (optional)
        The maximum number of iterations to perform
    tol: float (optional)
        The convergence criterion

    Returns
    -------
    s: float
        Distance in meters
    """

    a = globe_params["a"]
    b = globe_params["b"]
    f = (a - b) / a
    
    U1 = np.arctan((1-f) * np.tan(np.radians(lat1)))
    U2 = np.arctan((1-f) * np.tan(np.radians(lat2)))
    
    lamda = np.radians(lon2-lon1)
    converged = False
    it = 0
    
    while not converged and it < max_iter:

        cos_sigma = np.sin(U1) * np.sin(U2) + np.cos(U1) * np.cos(U2) \
                    * np.cos(lamda)
        sigma = np.arccos(cos_sigma)
        
        sin_alpha = np.cos(U1) * np.cos(U2) * np.sin(lamda) / np.sin(sigma)
        
        cos_2sigma_m = np.cos(sigma) \
                            - 2 * np.sin(U1) * np.sin(U2) / (1 - sin_alpha**2)
        C = (f / 16) * (1 - sin_alpha**2) \
                * (4 + f * (4 - 3 * (1 - sin_alpha**2)))
        lamda_old = lamda
        lamda = np.radians(lon2 - lon1) + (1 - C) * f * sin_alpha * (sigma 
               + C * np.sin(sigma) * (cos_2sigma_m \
                + C * np.cos(sigma) * (-1 + 2 * cos_2sigma_m**2)))
        
        if np.all(np.abs(lamda_old - lamda) / lamda_old < tol):
            converged = True
        it += 1
        
    u_sq =  (1 - sin_alpha**2) * (a**2 - b**2) / b**2
    A = 1 + u_sq / (16384) * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq *(-128 + u_sq * (74 - 47 * u_sq)))
    dsigma = B * np.sin(sigma) * (cos_2sigma_m \
                    + (B / 4) * (np.cos(sigma) * (-1 + 2 * cos_2sigma_m**2))
                    - (B / 6) * cos_2sigma_m * (-3 + 4 * np.sin(sigma)**2)\
                    * (-3 + 4 * cos_2sigma_m**2))
    
    s = b * A * (sigma - dsigma)
    return s 

def great_circle_intermediate_point(lon1, lat1, lon2, lat2, fraction):
    """
    Compute an intermediate point along the great circle path between
    (lon1, lat1) and (lon2, lat2), along 'fraction' of the distance.
    
    Parameters
    ----------
    lon1: float
        Longitude of first point, degrees
    lat1: float
        Latitude of first point, degrees
    lon2: float
        Longitude of second point, degrees
    lat2: float
        Latitude of second point, degrees    
    
    Returns
    -------
    lon_i: float
        Longitude of point, degrees
    lat_i: float
        Latitude of point, degrees
    """
    
    d = get_angular_distance(lon1, lat1, lon2, lat2)
    a = np.sin((1 - fraction) * d) / np.sin(d)
    b = np.sin(fraction * d) / np.sin(d)
    x = a * np.cos(np.radians(lat1)) * np.cos(np.radians(lon1)) \
            + b * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2))
    y = a * np.cos(np.radians(lat1)) * np.sin(np.radians(lon1)) \
            + b * np.cos(np.radians(lat2)) * np.sin(np.radians(lon2))
    z = a * np.sin(np.radians(lat1)) + b * np.sin(np.radians(lat2))
    
    lat_i = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lon_i = np.degrees(np.arctan2(y, x))
    
    return lon_i, lat_i