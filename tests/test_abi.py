import pytest

from CAP.abi import get_ABI_grid_locations

def test_get_ABI_grid_locations():

    # Values taken from ABI-L2 MCMIPF product
    col = 1111
    row = 1298
    x = -0.089627996
    y = 0.079156
    r, c = get_ABI_grid_locations(x, y)
    assert col == c
    assert row == r