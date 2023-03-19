import ctypes
import os
import numpy as np
from pathlib import Path

for file_dll in Path(os.path.dirname(__file__)).glob("libspectral_entropy*"):
    if file_dll.name.endswith(".dll"):
        c_lib = ctypes.CDLL(str(file_dll))
        break
    elif file_dll.name.endswith(".so"):
        c_lib = ctypes.CDLL(str(file_dll))
        break
    else:
        raise Exception("Cannot find libspectral_entropy.so or libspectral_entropy.dll")

func_apply_weight_to_intensity = c_lib.apply_weight_to_intensity
func_apply_weight_to_intensity.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]   # float_spec *spectrum, int spectrum_length

func_entropy_similarity = c_lib.entropy_similarity
func_entropy_similarity.restype = ctypes.c_float
func_entropy_similarity.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,   # float_spec *spectrum_a, int spectrum_a_len,
                                    ctypes.POINTER(ctypes.c_float), ctypes.c_int,   # float_spec *spectrum_b, int spectrum_b_len,
                                    ctypes.c_float, ctypes.c_int]                   # float ms2_da, bool spectra_is_preclean

func_unweighted_entropy_similarity = c_lib.unweighted_entropy_similarity
func_unweighted_entropy_similarity.restype = ctypes.c_float
func_unweighted_entropy_similarity.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,   # float_spec *spectrum_a, int spectrum_a_len,
                                               ctypes.POINTER(ctypes.c_float), ctypes.c_int,   # float_spec *spectrum_b, int spectrum_b_len,
                                               ctypes.c_float, ctypes.c_int]                   # float ms2_da, bool spectra_is_preclean

func_clean_spectrum = c_lib.clean_spectrum
func_clean_spectrum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),   # float_spec* spectrum, int* spectrum_length
                                ctypes.c_float, ctypes.c_float,                                 # float min_mz, float max_mz
                                ctypes.c_float,                                                 # float noise_threshold
                                ctypes.c_int,                                                   # int max_peak_num
                                ctypes.c_int,                                                   # bool normalize_intensity
                                ctypes.c_float]


def apply_weight_to_intensity(peaks):
    """
    Apply weight to intensity.
    """
    peaks = np.copy(peaks)
    func_apply_weight_to_intensity(peaks.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), peaks.shape[0])
    return peaks


def entropy_similarity(spec_query, spec_reference, ms2_da, spectra_is_preclean=False):
    spec_query = np.copy(spec_query)
    spec_reference = np.copy(spec_reference)
    return func_entropy_similarity(spec_query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), spec_query.shape[0],
                                   spec_reference.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), spec_reference.shape[0],
                                   ms2_da, spectra_is_preclean)


def unweighted_entropy_similarity(spec_query, spec_reference, ms2_da, spectra_is_preclean=False):
    spec_query = np.copy(spec_query)
    spec_reference = np.copy(spec_reference)
    return func_unweighted_entropy_similarity(spec_query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), spec_query.shape[0],
                                              spec_reference.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), spec_reference.shape[0],
                                              ms2_da, spectra_is_preclean)


def clean_spectrum(peaks, min_mz=0., max_mz=-1, noise_threshold=0.01, max_peak_num=-1, normalize_intensity=1, ms2_da=0.05):
    """
    Clean spectrum.
    """
    peaks = np.copy(peaks).astype(np.float32)
    peaks_length = ctypes.c_int(peaks.shape[0])
    func_clean_spectrum(peaks.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), peaks_length,
                        min_mz, max_mz,
                        noise_threshold,
                        max_peak_num,
                        normalize_intensity, ms2_da)
    peaks = peaks[:peaks_length.value]
    return peaks


