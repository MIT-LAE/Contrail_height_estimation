from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extension = Extension("CAP.interpolation", 
                ["CAP/interpolation.pyx"],
                libraries=[],
                include_dirs=[numpy.get_include()])

setup(ext_modules=cythonize(extension),

     package_data={'CAP': ['assets/*']},)