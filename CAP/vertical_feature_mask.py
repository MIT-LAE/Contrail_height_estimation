"""
Part of this code is based on that in the file of the same name in the repo:
https://github.com/ErickShepherd/modis_caliop_anomaly_analysis/

All definitions are based on the following source:
CALIPSO Data Products Catalog
Document No: PC-SCI-503 Release 4.97
"""
import itertools

import numpy as np
import pandas as pd

# VFM = Vertical feature mask
# Number of bits in the vertical feature mask.
VFM_INTEGER_SIZE = 16

# Table 92 in the CALIPSO Data Products Catalog
# concerns the first 3 bits in the vertical feature mask.
# Can be interpreted as "layer type"
FEATURE_TYPE = {
    0 : "invalid",
    1 : "clear air",
    2 : "cloud",
    3 : "tropospheric aerosol",
    4 : "stratospheric aerosol",
    5 : "surface",
    6 : "subsurface",
    7 : "no signal (totally attenuated)",
}

# Table 92 in the CALIPSO Data Products Catalog
# concerns the 4th and 5th bits in the vertical feature mask.
# Can be interpreted as "layer type quality assurance"
FEATURE_TYPE_QA = {
    0 : "none",
    1 : "low",
    2 : "medium",
    3 : "high",
}

# Table 92 in the CALIPSO Data Products Catalog
# concerns the 6th and 7th bits in the vertical feature mask.
# Can be interpreted as "cloud phase"
ICE_WATER_PHASE = {
    0 : "unknown / not determined",
    1 : "ice",
    2 : "water",
    3 : "oriented ice crystals",
}

# Table 92 in the CALIPSO Data Products Catalog
# concerns the 8th and 9th bits in the vertical feature mask.
# Can be interpreted as "cloud phase quality assurance"
ICE_WATER_PHASE_QA = {
    0 : "none",
    1 : "low",
    2 : "medium",
    3 : "high",
}

# Table 92 in the CALIPSO Data Products Catalog
# concerns bits 10-12 in the vertical feature mask.
# First index is the feature type, second index is the feature subtype.
FEATURE_SUBTYPE = {

    # clear air subtypes
    1 : {
        0 : "clear air",
    },

    # cloud subtypes
    2 : {
        0 : "low overcast, transparent",
        1 : "low overcas, opaque",
        2 : "transition stratocumulus",
        3 : "low, broken cumulus",
        4 : "altocumulus (transparent)",
        5 : "altostratus (opaque)",
        6 : "cirrus (transparent)",
        7 : "deep convective (opaque)",
    },

    # tropospheric aerosol sub-types
    3 : {
        0 : "not determined",
        1 : "clean marine",
        2 : "dust",
        3 : "polluted continental/smoke",
        4 : "clean continental",
        5 : "polluted dust",
        6 : "elevated smoke",
        7 : "dusty marine",
    },

    # stratospheric aerosol sub-types
    4 : {
        0 : "invalid",
        1 : "polar stratospheric aerosol",
        2 : "volcanic ash",
        3 : "sulfate",
        4 : "elevated smoke",
        5 : "unclassified",
        6 : "spare",
        7 : "spare",
    },
    
    # surface subtypes
    5 : {
        0 : "surface",
    },
    
    # subsurface subtypes
    6 : {
        0 : "subsurface",
    },
    
    # no signal subtypes
    7 : {
        0 : "no signal (totally attenuated)",
    },
}

# Table 92 in the CALIPSO Data Products Catalog
# concerns the 13th bit in the vertical feature mask.
# Can be interpreted as "feature subtype quality assurance"
CLOUD_AEROSOL_TYPE_QA = {
    0 : "not confident",
    1 : "confident",
}

# Table 92 in the CALIPSO Data Products Catalog
# concerns bits 14-16 in the vertical feature mask.
# Describes amount of horizontal averaging necessary for layer detection.
HORIZONTAL_AVERAGING = {
    0 : "not applicable",
    1 : "1/3 km",
    2 : "1 km",
    3 : "5 km",
    4 : "20 km",
    5 : "80 km",
}

def get_fcf_bitstring(feature_type, feature_type_qa, ice_water_phase,
    ice_water_phase_qa, feature_subtype, cloud_aerosol_type_qa,
    horizontal_averaging):
    """
    Returns the bitstring corresponding to a particular
    decoded feature classification flag.
    """
    
    def find_key(d, vq):
        try:
            return list(d.keys())[list(d.values()).index(vq)]
        # Value not in dictionary
        except ValueError:
            return None
    
    # Do `FEATURE_TYPE` first so that we know the FEATURE_SUBTYPE dict to
    # include
    integers = [find_key(FEATURE_TYPE, feature_type)]
    
    dicts = [FEATURE_TYPE_QA, ICE_WATER_PHASE, ICE_WATER_PHASE_QA,
            FEATURE_SUBTYPE[integers[0]], CLOUD_AEROSOL_TYPE_QA,
            HORIZONTAL_AVERAGING]
    vals = [feature_type_qa, ice_water_phase, ice_water_phase_qa,
            feature_subtype, cloud_aerosol_type_qa, horizontal_averaging]
    for (d, vq) in zip(dicts, vals):
        integers.append(find_key(d, vq))

    # Reverse integers so that the first element is the first bit
    integers = integers[::-1]

    # Number of bits for each 'section' of the bitstring.
    # Based on CALIPSO Data Products Catalog Table 92
    lengths = [3, 1, 3, 2, 2, 2, 3]

    bitstring = ""
    for integer, length in zip(integers, lengths):
        
        # Convert integer to binary, remove '0b' using [2:] and pad with
        # zeros to the correct length.
        bitstring += bin(integer)[2:].zfill(length)
        
    return bitstring
    
    
def get_cirrus_fcf_integers():
    """
    This function returns the integers corresponding to cirrus clouds within
    the CALIOP L2 Feature_Classification_Flags dataset.
    """
    
    ice_water_phase = ["ice", "oriented ice crystals"]
    cloud_sub_type = ["cirrus (transparent)", "deep convective (opaque)"]
    cloud_aerosol_type_qa = ["not confident", "confident"]
    horizontal_averaging = ["5 km", "1 km"]
    
    integers = []
    for comb in itertools.product(ice_water_phase, cloud_sub_type,
                cloud_aerosol_type_qa, horizontal_averaging):
        
        bstring = get_fcf_bitstring("cloud", "high", comb[0], "high",
                                    comb[1], comb[2], comb[3])
        
        integers.append(int(bstring,2))
        
    return integers