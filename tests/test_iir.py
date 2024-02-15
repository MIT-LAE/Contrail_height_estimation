import numpy as np

from CAP.iir import planck_function

def test_planck_function():
    # Test based on wikipedia plot of Planck function (peak of Planck curves)
    T = np.array([5000., 4000., 3000.]) # Temperature, K
    lamdas = np.array([0.6, 0.66, 1.0]) # Wavelength, micron
    values = np.array([12.5e12, 4.2e12, 1.e12]) # W/m^2/steradian/m

    np.testing.assert_allclose(planck_function(T, lamdas * 1e-6), values,
            rtol=0.1)