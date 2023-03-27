import numpy as np
from typing import Union
import scipy.stats

from .tools import clean_spectrum
from .tools_match_spectra import match_spectrum_output_number
from .spectral_similarity_simple import entropy_distance


def merge_matched_id_array(peaks_match_array_1, peaks_match_array_2, peaks_2):
    for i in range(len(peaks_match_array_1)):
        if peaks_match_array_2[i, 1] >= 0:
            if peaks_match_array_1[i, 1] < 0 or \
                    peaks_2[peaks_match_array_2[i, 1], 1] > peaks_2[peaks_match_array_1[i, 1], 1]:
                peaks_match_array_1[i, 1] = peaks_match_array_2[i, 1]
    return peaks_match_array_1


def calculate_entropy_similarity(peaks_match_array, peaks_a, peaks_b):
    p_a = np.copy(peaks_a[:, 1])
    p_b = np.copy(peaks_b[:, 1])
    p_ba = np.zeros_like(p_b)
    for a, b in peaks_match_array:
        if b >= 0:
            p_ba[b] += p_a[a]
            p_a[a] = 0

    spectral_distance = entropy_distance(np.concatenate([p_a, p_ba]),
                                         np.concatenate([np.zeros_like(p_a), p_b]))
    spectral_similarity = 1 - spectral_distance / np.log(4)
    return spectral_similarity


def calculate_spectral_entropy(peaks):
    return scipy.stats.entropy(peaks[:, 1])


def similarity(peaks_query: Union[list, np.ndarray], peaks_library: Union[list, np.ndarray],
               precursor_mz_delta: float, method: str = "bonanza",
               ms2_ppm: float = None, ms2_da: float = None,
               clean_spectra: bool = True) -> float:
    """
    Calculate the similarity between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :param peaks_query:
    :param peaks_library:
    :param precursor_mz_delta: query precursor mz - library precursor mz
    :param method: Supported methods: "bonanza"
    :param ms2_ppm:
    :param ms2_da:
    :param clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :return: Similarity between two spectra
    """
    peaks_query = np.asarray(peaks_query, dtype=np.float32)
    peaks_library = np.asarray(peaks_library, dtype=np.float32)
    if clean_spectra:
        peaks_query = clean_spectrum(peaks_query, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        peaks_library = clean_spectrum(peaks_library, ms2_ppm=ms2_ppm, ms2_da=ms2_da)

    # Calculate similarity
    if peaks_query.shape[0] > 0 and peaks_library.shape[0] > 0:
        peaks_matched_ori = match_spectrum_output_number(spec_a=peaks_query,
                                                         spec_b=peaks_library,
                                                         ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        peaks_query[:, 0] -= precursor_mz_delta
        peaks_matched_shift = match_spectrum_output_number(spec_a=peaks_query,
                                                           spec_b=peaks_library,
                                                           ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        peaks_query[:, 0] += precursor_mz_delta

        peaks_matched_id_array = merge_matched_id_array(peaks_matched_ori, peaks_matched_shift, peaks_library)

        if np.sum(peaks_matched_id_array[:, 1] >= 0) == 0:
            spectral_similarity = 0
        else:
            spectral_similarity = calculate_entropy_similarity(peaks_matched_id_array, peaks_query, peaks_library)
        return spectral_similarity
    return 0.
