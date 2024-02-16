"""
Based on file of the same name in:
https://github.com/ErickShepherd/modis_caliop_anomaly_analysis/

I have switched around keys of aerosol and cloud categories in the
FEATURE_SUBTYPE dictionary, as I believe these were incorrect in the original
file. I have also increased the number of comments.

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
    
    ice_water_phase = ["ice", "oriented ice crystals"]
    cloud_sub_type = ["cirrus (transparent)", "deep convective (opaque)"]
    cloud_aerosol_type_qa = ["not confident", "confident"]
    horizontal_averaging = ["5 km", "1 km"]
    
    integers = []
    for comb in itertools.product(ice_water_phase, cloud_sub_type, cloud_aerosol_type_qa, horizontal_averaging):
        
        bstring = get_fcf_bitstring("cloud", "high", comb[0], "high", comb[1], comb[2], comb[3])
        
        integers.append(int(bstring,2))
        
    return integers
        
    

def digital_to_binary(a):
    a = np.asarray(a)
    
    # Ensures the result is a big-endian of the appropriate size.
    a = a.astype(f">u{a.itemsize}")
        
    bits = a[np.newaxis].view(np.uint8).reshape((*a.shape, a.itemsize))
    bits = np.unpackbits(bits, axis = -1)
        
    return bits


def binary_to_digital(a):

    digits = np.sum(2 ** np.arange(a.shape[-1])[::-1] * a, axis = -1)
    
    return digits


def slice_binary_array(a):
    
    # Transposition to swap the last and first axes for slicing.
    feature_type         = a.T[-1:-4:-1][::-1].T
    feature_type_qa      = a.T[-4:-6:-1][::-1].T
    ice_water_phase      = a.T[-6:-8:-1][::-1].T
    ice_water_phase_qa   = a.T[-8:-10:-1][::-1].T
    feature_subtype      = a.T[-10:-13:-1][::-1].T
    feature_subtype_qa   = a.T[-13:-14:-1][::-1].T
    horizontal_averaging = a.T[-14:-17:-1][::-1].T
    
    vfm = [feature_type,
           feature_type_qa,
           ice_water_phase,
           ice_water_phase_qa,
           feature_subtype,
           feature_subtype_qa,
           horizontal_averaging]
    
    return vfm


def extract_features(a):
    
    """
    
    Extracts the selected bits from the binary vertical feature mask and
    converts them into a base 10 integer.
    
    The substring slicing for this function counts from 1 (as opposed to 0) to
    match the convention used by the CALIPSO team. This convention is
    specified in the following source:
        
        PC-SCI-503: CALIPSO Data Products Catalog (Version 3.2)
         - Table 44: Feature Classification Flag Definition
    
    """
    
    # Converts the input data into its constituent bits.
    bits = digital_to_binary(a)
    
    # Slices the array bits for classification.
    vfm = slice_binary_array(bits)
    
    # Converts the array slices from binary to digital.
    for index in range(len(vfm)):
        
        vfm[index] = binary_to_digital(vfm[index])
    
    # Moves the result axis to the end so that the output array axis order
    # matches that of the input array.
    vfm = np.moveaxis(vfm, 0, -1)
    
    return vfm


def _extract_feature(binary_vfm_cell_value, start, stop):
    
    """
    
    Extracts the selected bits from the binary vertical feature mask and
    converts them into a base 10 integer.
    
    The substring slicing for this function counts from 1 (as opposed to 0) to
    match the convention used by the CALIPSO team. This convention is
    specified in the following source:
        
        PC-SCI-503: CALIPSO Data Products Catalog (Version 3.2)
         - Table 44: Feature Classification Flag Definition
    
    """
    
    substring = binary_vfm_cell_value[::-1][start - 1:stop][::-1]
    integer   = int(substring, 2)
    
    return integer


def _classify_feature_flags(vfm_cell_value):
    
    """
    
    Classifies CALIPSO 16-bit integer vertical feature mask classification
    flags based on their bitwise values.
    
    """
    
    vfm_cell_value = np.binary_repr(vfm_cell_value, VFM_INTEGER_SIZE)
    
    vfm = {
        "feature_type"         : _extract_feature(vfm_cell_value,  1,  3),
        "feature_type_qa"      : _extract_feature(vfm_cell_value,  4,  5),
        "ice_water_phase"      : _extract_feature(vfm_cell_value,  6,  7),
        "ice_water_phase_qa"   : _extract_feature(vfm_cell_value,  8,  9),
        "feature_subtype"      : _extract_feature(vfm_cell_value, 10, 12),
        "feature_subtype_qa"   : _extract_feature(vfm_cell_value, 13, 13),
        "horizontal_averaging" : _extract_feature(vfm_cell_value, 14, 16),
    }
        
    vfm = pd.Series(vfm, dtype = np.object_)
        
    return vfm


def _interpret_feature_flags(vfm):
    
    """
    
    Applies a physical interpretation to the classified CALIPSO feature flags.
    The physical interpretation is supplied by the following source:
    
        PC-SCI-503: CALIPSO Data Products Catalog (Version 3.2)
         - Table 44: Feature Classification Flag Definition
    
    """
    
    vfm = _classify_feature_flags(vfm)
    
    feature_type = \
        FEATURE_TYPE[vfm["feature_type"]]
    
    feature_type_qa = \
        FEATURE_TYPE_QA[vfm["feature_type_qa"]]
    
    ice_water_phase = \
        ICE_WATER_PHASE[vfm["ice_water_phase"]]
    
    ice_water_phase_qa = \
        ICE_WATER_PHASE_QA[vfm["ice_water_phase_qa"]]
        
    feature_subtype = \
        FEATURE_SUBTYPE[vfm["feature_type"]][vfm["feature_subtype"]]
        
    feature_subtype_qa = \
        CLOUD_AEROSOL_TYPE_QA[vfm["feature_subtype_qa"]]
    
    horizontal_averaging = \
        HORIZONTAL_AVERAGING[vfm["horizontal_averaging"]]
    
    vfm = pd.Series(dtype = np.object_)
    
    vfm["feature_type"]         = feature_type
    vfm["feature_type_qa"]      = feature_type_qa
    vfm["ice_water_phase"]      = ice_water_phase
    vfm["ice_water_phase_qa"]   = ice_water_phase_qa
    vfm["feature_subtype"]      = feature_subtype
    vfm["feature_subtype_qa"]   = feature_subtype_qa
    vfm["horizontal_averaging"] = horizontal_averaging
        
    return vfm


# Vectorizations.
classify_feature_flags  = np.vectorize(_classify_feature_flags)
interpret_feature_flags = np.vectorize(_interpret_feature_flags)

# Alias definitions.
classify_vfm  = classify_feature_flags
interpret_vfm = interpret_feature_flags

