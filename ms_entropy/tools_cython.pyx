import numpy as np
cimport numpy as np

ctypedef np.float32_t float32
ctypedef np.int64_t int_64
ctypedef np.int8_t int_8
ctypedef np.uint32_t uint_32
from libc.math cimport log2,log,pow


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


cpdef apply_weight_to_intensity(float32[:,::1] spectrum):
    """
    Apply the weight to the intensity.
    The spectrum is a 2D array like: [[m/z, intensity], [m/z, intensity], ...]
    """
    cdef double entropy=spectral_entropy(spectrum)
    cdef double weight, intensity_sum
    cdef float32[:,::1] spectrum_weighted = np.copy(spectrum)
    if entropy<3:
        weight = 0.25 + 0.25 * entropy
        intensity_sum = 0.
        for i in range(spectrum_weighted.shape[0]):
            spectrum_weighted[i,1] = pow(spectrum_weighted[i,1],weight)
            intensity_sum += spectrum_weighted[i,1]

        if intensity_sum>0:
            for i in range(spectrum_weighted.shape[0]):
                spectrum_weighted[i,1] /= intensity_sum
            
    return spectrum_weighted

cpdef double spectral_entropy(float32[:,::1] spectrum) nogil:
    """
    Compute the spectral entropy of a spectrum. The intensity need to be pre-normalized, the function will not do it.
    """
    cdef double entropy=0.
    cdef float32 intensity
    for i in range(spectrum.shape[0]):
        intensity=spectrum[i,1]
        if intensity>0:
            entropy+=-intensity*log(intensity)
    return entropy


cpdef centroid_spectrum(float32[:,::1] spectrum,ms2_ppm=None, ms2_da=None):
    """
    Calculate centroid spectrum from a spectrum.
    At least one of the ms2_ppm or ms2_da need be not None. If both ms2_da and ms2_ppm is given, ms2_da will be used.

    :param spectrum: The spectrum should be a 2D array with the first dimension being the m/z values and 
                     the second dimension being the intensity values.
                     The spectrum need to be sorted by m/z.
                     The spectrum should be in C order.
    :param ms2_ppm: the mass accuracy in ppm.
    :param ms2_da: the mass accuracy in Da.
    """
    if ms2_da is None:
        ms2_da = -1.
    else:
        ms2_ppm = -1.

    # Check whether the spectrum needs to be centroided or not.
    cdef int need_centroid = check_centroid_c(spectrum, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
    while need_centroid:
        spectrum = centroid_spectrum_c(spectrum, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        need_centroid = check_centroid_c(spectrum, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
    return np.asarray(spectrum)

cdef centroid_spectrum_c(float32[:,::1] spec,double ms2_ppm, double ms2_da):
    """
    Calculate centroid spectrum from a spectrum.
    At least one of the ms2_ppm or ms2_da need be not None. If both ms2_da and ms2_ppm is given, ms2_da will be used.

    :param spec: the spectrum should be a 2D array with the first dimension being the m/z values and 
                    the second dimension being the intensity values.
                     The spectrum should be in C order.
    :param ms2_ppm: the mass accuracy in ppm.
    :param ms2_da: the mass accuracy in Da.
    """
    cdef int_64[:] intensity_order = np.argsort(spec[:, 1])
    cdef float32[:,::1] spec_new=np.zeros((spec.shape[0],2),dtype=np.float32,order='C')
    cdef int spec_new_i=0

    cdef double mz_delta_allowed
    cdef Py_ssize_t idx,x
    cdef Py_ssize_t i_left,i_right
    cdef float32 mz_delta_left,mz_delta_right,intensity_sum,intensity_weighted_sum

    with nogil:
        for x in range(intensity_order.shape[0]-1, -1, -1):
            idx = intensity_order[x]
            if ms2_da >= 0:
                mz_delta_allowed = ms2_da
            else:
                mz_delta_allowed = ms2_ppm * 1e-6 * spec[idx, 0]

            if spec[idx, 1] > 0:
                # Find left board for current peak
                i_left = idx - 1
                while i_left >= 0:
                    mz_delta_left = spec[idx, 0] - spec[i_left, 0]
                    if mz_delta_left <= mz_delta_allowed:
                        i_left -= 1
                    else:
                        break
                i_left += 1

                # Find right board for current peak
                i_right = idx + 1
                while i_right < spec.shape[0]:
                    mz_delta_right = spec[i_right, 0] - spec[idx, 0]
                    if mz_delta_right <= mz_delta_allowed:
                        i_right += 1
                    else:
                        break

                # Merge those peaks
                intensity_sum = 0
                intensity_weighted_sum = 0
                for i_cur in range(i_left, i_right):
                    intensity_sum += spec[i_cur, 1]
                    intensity_weighted_sum += spec[i_cur, 0]*spec[i_cur, 1]

                spec_new[spec_new_i, 0] = intensity_weighted_sum / intensity_sum
                spec_new[spec_new_i, 1] = intensity_sum
                spec_new_i += 1
                spec[i_left:i_right, 1] = 0

    spec_result = spec_new[:spec_new_i,:]
    spec_result=np.array(spec_result)[np.argsort(spec_result[:,0])]
    return spec_result


cdef int check_centroid_c(float32[:,::1] spectrum,double ms2_ppm, double ms2_da):
    """
    Check whether the spectrum needs to be centroided or not.
    At least one of the ms2_ppm or ms2_da need be not None. If both ms2_da and ms2_ppm is given, ms2_da will be used.
    
    :param spectrum: the spectrum should be a 2D array with the first dimension being the m/z values and 
                     the second dimension being the intensity values.
                     The spectrum need to be sorted by m/z.
                     The spectrum should be in C order.
    :param ms2_ppm: the mass accuracy in ppm.
    :param ms2_da: the mass accuracy in Da.
    """
    if spectrum.shape[0]<=1:
        return 0

    if ms2_da>0:
        # Use Da accuracy
        for i in range(1, spectrum.shape[0]):
            if spectrum[i, 0]-spectrum[i-1, 0] < ms2_da:
                return 1
    else:
        # Use ppm accuracy
        for i in range(1, spectrum.shape[0]):
            if spectrum[i, 0]-spectrum[i-1, 0] < spectrum[i, 0]*ms2_ppm*1e-6:
                return 1
    return 0
