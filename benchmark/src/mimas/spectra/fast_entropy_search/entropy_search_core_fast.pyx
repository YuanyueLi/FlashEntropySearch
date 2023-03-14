import numpy as np
cimport numpy as np
from libc.math cimport log2,log,pow
from cython.parallel import prange

ctypedef np.float32_t float32
ctypedef np.int8_t int_8
ctypedef np.uint32_t uint_32
ctypedef np.int64_t int_64


cpdef void entropy_similarity_search_fast(int_64 product_mz_idx_min, int_64 product_mz_idx_max,
                                    float32 intensity, float32[:] mixed_spectra_entropy,
                                    const float32[:] library_peaks_intensity, const uint_32[:] library_spec_idx_array,
                                    int search_type, int_64 search_spectra_idx_min, int_64 search_spectra_idx_max, const int_8[:] search_array) nogil:
    """
    The mixed_spectra_entropy will be modified in this function.
    search_type is 0: search all spectra.
    search_type is 1: search spectra in the range [search_spectra_idx_min, search_spectra_idx_max).
    search_type is 2: search spectra in the array search_array with entry equals 1, the length of search_array should be equal to the self.total_spectra_num

    Note: the intensity here should be half of the original intensity.
    """
    cdef uint_32 library_spec_idx
    cdef float32 library_peak_intensity, intensity_ab
    cdef float32 intensity_xlog2x = intensity * log2(intensity)

    for idx in range(product_mz_idx_min, product_mz_idx_max):
        library_spec_idx = library_spec_idx_array[idx]
        if (search_type == 0) or \
                (search_type == 1 and search_spectra_idx_min <= library_spec_idx and library_spec_idx < search_spectra_idx_max) or \
                (search_type == 2 and search_array[library_spec_idx]):
            # Match this peak
            library_peak_intensity = library_peaks_intensity[idx]
            intensity_ab = intensity + library_peak_intensity

            mixed_spectra_entropy[library_spec_idx] += \
                intensity_ab * log2(intensity_ab) - \
                intensity_xlog2x - \
                library_peak_intensity * log2(library_peak_intensity)

cpdef void entropy_similarity_search_inplace(
    int_64 product_mz_idx_max, float32 intensity, float32[:] library_peaks_intensity) nogil:
    """
    The mixed_spectra_entropy will be modified in this function.
    search_type is 0: search all spectra.
    search_type is 1: search spectra in the range [search_spectra_idx_min, search_spectra_idx_max).
    search_type is 2: search spectra in the array search_array with entry equals 1, the length of search_array should be equal to the self.total_spectra_num

    Note: the intensity here should be half of the original intensity.
    """
    cdef float32 library_peak_intensity, intensity_ab
    cdef float32 intensity_xlog2x = intensity * log2(intensity)

    for idx in range(product_mz_idx_max):
        # Match this peak
        library_peak_intensity = library_peaks_intensity[idx]
        intensity_ab = intensity + library_peak_intensity

        library_peaks_intensity[idx] = \
            intensity_ab * log2(intensity_ab) - \
            intensity_xlog2x - \
            library_peak_intensity * log2(library_peak_intensity)



cpdef void entropy_similarity_search_fast_mp(int_64 product_mz_idx_min, int_64 product_mz_idx_max,
                                    float32 intensity, float32[:] mixed_spectra_entropy,
                                    const float32[:] library_peaks_intensity, const uint_32[:] library_spec_idx_array,
                                    int search_type, int_64 search_spectra_idx_min, int_64 search_spectra_idx_max, const int_8[:] search_array) nogil:
    """
    The mixed_spectra_entropy will be modified in this function.
    search_type is 0: search all spectra.
    search_type is 1: search spectra in the range [search_spectra_idx_min, search_spectra_idx_max).
    search_type is 2: search spectra in the array search_array with entry equals 1, the length of search_array should be equal to the self.total_spectra_num

    Note: the intensity here should be half of the original intensity.
    """
    cdef uint_32 library_spec_idx
    cdef float32 library_peak_intensity, intensity_ab
    cdef float32 intensity_xlog2x = intensity * log2(intensity)
    cdef int_64 idx

    for idx in prange(product_mz_idx_min, product_mz_idx_max,nogil=True):
        library_spec_idx = library_spec_idx_array[idx]
        if (search_type == 0) or \
                (search_type == 1 and search_spectra_idx_min <= library_spec_idx and library_spec_idx < search_spectra_idx_max) or \
                (search_type == 2 and search_array[library_spec_idx]):
            # Match this peak
            library_peak_intensity = library_peaks_intensity[idx]
            intensity_ab = intensity + library_peak_intensity

            mixed_spectra_entropy[library_spec_idx] += \
                intensity_ab * log2(intensity_ab) - \
                intensity_xlog2x - \
                library_peak_intensity * log2(library_peak_intensity)

