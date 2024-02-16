import numpy as np
import pytest

from CAP.vertical_feature_mask import get_fcf_bitstring


def test_get_fcf_bitstring():
    """
    This test considers the CALIOP L2 5 km cloud layer product
    CAL_LID_L2_05kmCLay-Standard-V4-20.2018-08-08T18-58-40ZD
    for the subset of the domain starting at latitude 47.61
    and ending at latitude 53.60.
    
    The CALIOP LARC website gives the interpreted vertical feature mask
    at (https://www-calipso.larc.nasa.gov/products/lidar/browse_images/
    show_v4_detail.php?s=production&v=V4-10&browse_date=2018-08-08
    &orbit_time=18-58-40&page=3
    &granule_name=CAL_LID_L1-Standard-V4-10.2018-08-08T18-58-40ZD.hdf)
    
    The first feature (i.e. at the min latitude) is "purple" in the vertical
    feature mask image corresponding to a cloud identified with low confidence.
    
    The second feature right after it is is "cyan" in the vertical feature
    mask image corresponding to a cloud identified with higher confidence.
    
    The vertical feature mask feature classification flags here are
    extracted from the L2 5km layer product directly. They correspond to
    the highest altitude feature found in a particular 5 km layer.

    To reproduce these feature classification flag values, you can use the
    following code.
    ```python
    from CAP.caliop import CALIOP

    # Change this!
    path = `path/to/CAL_LID_L1-Standard-V4-10.2018-08-08T18-58-40ZD.hdf"`
    ca = CALIOP(path)

    fcf = ca.get("Feature_Classification_Flags")
    lat = ca.get("Latitude")
    lon = ca.get("Longitude")

    min_lat = 47.61
    max_lat = 53.60

    # First column correspond to layer start
    # Last column corresponds to layer end
    row_idx = (lat[:,0] >= min_lat) * (lat[:,-1] <= max_lat)

    lat = lat[row_idx,:]
    lon = lon[row_idx,:]

    # Row is layer dimension, col is vertical dimension
    fcf = fcf[row_idx,:]
    
    first_fcf = int(fcf[0,0])

    # First 8 layers are occupied by the first feature
    second_fcf = int(fcf[9,0]) 
    ```
    """
    # From the CALIOP L2 5 km cloud layer product
    first_fcf = 44034
    second_fcf = 28090
    
    # Convert to bitstrings
    first_fcf_bitstring = bin(first_fcf)[2:].zfill(16)
    second_fcf_bitstring = bin(second_fcf)[2:].zfill(16)

    # Now test the get_fcf_bitstring function
    # we can only test the correctness of subsets of the bitstring
    # as the imagery at the link in the docstring does not give all confidence
    # levels.
    feature_type = "cloud" # Magenta/purple in image
    feature_type_qa = "none" # Guess, cannot verify as not given
    horizontal_averaging = "80 km" # Dark blue in image
    ice_water_phase = "unknown / not determined" # Red in image
    ice_water_phase_qa = "none" # Guess, cannot verify
    feature_subtype = "cirrus (transparent)" # Gray in image
    feature_subtype_qa = "not confident" # Guess, cannot verify

    bitstr = get_fcf_bitstring(feature_type, feature_type_qa, ice_water_phase,
                    ice_water_phase_qa, feature_subtype, feature_subtype_qa,
                    horizontal_averaging)
    
    # Test the first 3 bits, feature type
    assert bitstr[-3:] == first_fcf_bitstring[-3:]
    # Test bits 6 and 7, ice water phase
    assert bitstr[-7:-5] == first_fcf_bitstring[-7:-5]
    # Test bits 10-12, feature subtype
    assert bitstr[-12:-9] == first_fcf_bitstring[-12:-9]
    # test bits 14-16, horizontal averaging
    assert bitstr[:2] == first_fcf_bitstring[:2]

    # Same for second bitstring
    feature_type = "cloud" # Cyan in image
    feature_type_qa = "none" # Guess, cannot verify as not given
    horizontal_averaging = "5 km" # Yellow in image
    ice_water_phase = "ice" # White in image
    ice_water_phase_qa = "none" # Guess, cannot verify
    feature_subtype = "cirrus (transparent)" # Gray in image
    feature_subtype_qa = "not confident" # Guess, cannot verify

    bitstr = get_fcf_bitstring(feature_type, feature_type_qa, ice_water_phase,
                    ice_water_phase_qa, feature_subtype, feature_subtype_qa,
                    horizontal_averaging)
    
    # Test the first 3 bits, feature type
    assert bitstr[-3:] == second_fcf_bitstring[-3:]
    # Test bits 6 and 7, ice water phase
    assert bitstr[-7:-5] == second_fcf_bitstring[-7:-5]
    # Test bits 10-12, feature subtype
    assert bitstr[-12:-9] == second_fcf_bitstring[-12:-9]
    # test bits 14-16, horizontal averaging
    assert bitstr[:2] == second_fcf_bitstring[:2]




