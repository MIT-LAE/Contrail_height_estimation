import numpy as np
from src.geometry import *
import pytest


def test_get_bearing():

    lon0 = 45.
    lat0 = 35.
    lon1 = 135.
    lat1 = 35.

    bearing = get_bearing(lon0, lat0, lon1, lat1)
    assert pytest.approx(bearing, rel=0.01) == 60.

    # Reverse
    bearing = get_bearing(lon1, lat1, lon0, lat0)
    assert pytest.approx((bearing + 180. % 360.), rel=0.01) == 120.


def test_get_haversine():

        lon0 = 0
        lat0 = 0
        lon1 = 180
        lat1 = 0 
        result = get_haversine(lon0, lat0, lon1, lat1)
        assert pytest.approx(result, 1) == np.pi * RADIUS_EARTH


def test_greatcirclepath():
    path = GreatCirclePath(0, 0, 0, 90)
    dist = path.get_crosstrack_distance(0, 0)
    assert pytest.approx(dist, 0.1) == 0.0

    path = GreatCirclePath(0, 0, 180, 0)
    dist = path.get_crosstrack_distance(0, 90)
    assert pytest.approx(np.abs(dist), 1) == 0.5*np.pi * RADIUS_EARTH


def test_ECEF2geodetic():

    # Map ECEF x axis to geodetic
    x = np.array([1.0, 0.0, 0.0]).reshape(3,1)
    lon, lat = ECEF2geodetic(x)
    assert pytest.approx(lon) == 0.
    assert pytest.approx(lat) == 0.

    # Map ECEF y axis to geodetic
    x = np.array([0.0, 1.0, 0.0]).reshape(3,1)
    lon, lat = ECEF2geodetic(x)
    assert pytest.approx(lon) == 90.
    assert pytest.approx(lat) == 0.

    # Map ECEF z axis to geodetic
    x = np.array([0.0, 0.0, 1.0]).reshape(3,1)
    lon, lat = ECEF2geodetic(x)
    assert pytest.approx(lon) == 0.
    assert pytest.approx(lat) == 90.

    
def test_geodesic_distance():
    ans = geodesic_distance(-50, 15, -50, 20)
    assert pytest.approx(ans) == 553376.665

def test_parallax_correction_vicente_forward():

    # Order of magnitude test against values
    # from https://cimss.ssec.wisc.edu/satellite-blog/archives/35400

    # Orlando
    lon = -81.379234
    lat = 28.538336
    lon_c, lat_c = parallax_correction_vicente_forward(lon, lat, 4.5e3)
    assert pytest.approx(geodesic_distance(lon, lat, lon_c, lat_c), rel=1e-1) == 3000.

    # Seattle
    lon = -122.335
    lat = 47.608013
    lon_c, lat_c = parallax_correction_vicente_forward(lon, lat, 4.5e3)
    assert pytest.approx(geodesic_distance(lon, lat, lon_c, lat_c), rel=1e-1) == 13000.


def test_get_ABI_grid_locations():

    # Values taken from ABI-L2 MCMIPF product
    col = 1111
    row = 1298
    x = -0.089627996
    y = 0.079156
    r, c = get_ABI_grid_locations(x, y)
    assert col == c
    assert row == r


def test_great_circle_intermediate_point():
    
    # Example from http://www.movable-type.co.uk/scripts/latlong.html
    lon, lat = great_circle_intermediate_point(45., 35., 135., 35., 0.5)
    
    assert pytest.approx(lon) == 90.
    assert pytest.approx(lat, rel=1e-1) == 45.
    
    # Another one
    mid_lon = -50.5813888889
    mid_lat = 34.
    lon, lat = great_circle_intermediate_point(-33., 45., -66., 23., 0.5)
    assert pytest.approx(lon, rel=1e-1) == mid_lon
    assert pytest.approx(lat, rel=1e-1) == mid_lat


def test_parallax_correction_vicente():

    # Exactly below satellite, no parallax
    lon = -75.2
    lat = 0.
    h = 10e3
    lon_c, lat_c = parallax_correction_vicente_forward(lon, lat, h)

    assert pytest.approx(lon) == lon_c
    assert pytest.approx(lat) == lat_c


    # Zero cloud height
    lon = -75.2
    lat = 10.0
    h = 0
    lon_c, lat_c = parallax_correction_vicente_forward(lon, lat, h)

    assert pytest.approx(lon,rel=1e-4) == lon_c
    assert pytest.approx(lat,rel=1e-4) == lat_c
