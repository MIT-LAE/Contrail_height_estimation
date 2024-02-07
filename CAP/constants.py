# Geodetic Reference System 1980 (GRS80) parameters, see 
# "Geodetic Reference System 1980" by Moritz, 1980
GRS80_PARAMS = {"b": 6356752.314140347, "a": 6378137.} # Both in meters

# Mean radius of the Earth based on GRS80 system
RADIUS_EARTH = 0.5 * (GRS80_PARAMS["a"] + GRS80_PARAMS["b"])# meters