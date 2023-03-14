from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
# sys.argv[1:] = ['build_ext']

os.environ['CFLAGS'] = '-O3 -Wno-cpp -Wno-unused-function'

setup(
    name='Entropy similarity',
    ext_modules=cythonize([
        Extension('flash_entropy.libspectral_entropy',
                  [r"flash_entropy/CleanSpectrum.c",
                   r"flash_entropy/SpectralEntropy.c"]),

        Extension('flash_entropy.entropy_search_core_fast',
                  [r"flash_entropy/entropy_search_core_fast.pyx"]),
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
