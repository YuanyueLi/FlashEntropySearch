from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
sys.argv[1:] = ['build_ext']

os.environ['CFLAGS'] = '-O3 -Wno-cpp -Wno-unused-function'

setup(
    name='Entropy similarity',
    ext_modules=cythonize([
        Extension('mimas.spectra.spectral_entropy_c.libspectral_entropy',
                  [r"mimas/spectra/spectral_entropy_c/CleanSpectrum.c",
                   r"mimas/spectra/spectral_entropy_c/SpectralEntropy.c"]),

        Extension('mimas.spectra.fast_entropy_search.entropy_search_core_fast',
                  [r"mimas/spectra/fast_entropy_search/entropy_search_core_fast.pyx"]),
    ],
        annotate=False,
        compiler_directives={
        'language_level': "3",
        'cdivision': True,
        'boundscheck': False,  # turn off bounds-checking for entire function
        'wraparound': False  # turn off negative index wrapping for entire function
    }),
    include_dirs=[np.get_include()]
)

# python setup.py build_ext
