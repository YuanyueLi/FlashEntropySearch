# cython: infer_types=True

import numpy as np
cimport numpy as np
from libc.math cimport log2,log,pow

ctypedef np.float32_t float32
ctypedef np.int64_t int_64

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


cpdef double intensity_entropy(float32[:] intensity) nogil:
    """
    Compute the spectral entropy of a spectrum. The intensity need to be pre-normalized, the function will not do it.
    """
    cdef double entropy=0.
    cdef float32 intensity_cur
    for i in range(intensity.shape[0]):
        intensity_cur=intensity[i]
        if intensity_cur>0:
            entropy+=-intensity_cur*log(intensity_cur)
    return entropy


cpdef double spectral_entropy_log2(float32[:,::1] spectrum) nogil:
    """
    Compute the spectral entropy of a spectrum.
    """
    cdef double entropy=0.
    cdef float32 intensity
    for i in range(spectrum.shape[0]):
        intensity=spectrum[i,1]
        if intensity>0:
            entropy+=-intensity*log2(intensity)
    return entropy


cpdef double unweighted_entropy_similarity(float32[:,::1] spectrum_a,float32[:,::1] spectrum_b,ms2_ppm=None, ms2_da=None):
    """
    Calculate the unweighted entropy similarity between two spectra.
    Both spectra need to be centroided, sorted by m/z and their intensities need to be normalized to sum to 1.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    """
    if ms2_da is not None:
        return unweighted_entropy_similarity_c(spectrum_a, spectrum_b, ms2_ppm=-1, ms2_da=ms2_da)
    else:
        return unweighted_entropy_similarity_c(spectrum_a,spectrum_b,ms2_ppm=ms2_ppm,ms2_da=-1)



cdef double unweighted_entropy_similarity_c(float32[:,:] spec_a,float32[:,:] spec_b,double ms2_ppm,double ms2_da):
    """
    Calculate the unweighted entropy similarity between two spectra.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :return: list. Each element in the list is a list contain three elements:
                              m/z, intensity from spec 1; intensity from spec 2.
    """
    cdef int a = 0
    cdef int b = 0
    cdef int i

    cdef float32 peak_b_int = 0.
    cdef float32 mass_delta_da

    cdef double entropy_merged = 0.
    cdef double entropy_a = 0.
    cdef double entropy_b = 0.
    cdef double peak_cur = 0.

    while a < spec_a.shape[0] and b < spec_b.shape[0]:
        if ms2_ppm > 0:
            ms2_da = ms2_ppm * -1e6 * spec_a[a,0]
        mass_delta_da = spec_a[a, 0] - spec_b[b, 0]

        if mass_delta_da < -ms2_da:
            # Peak only existed in spec a.
            entropy_a += -spec_a[a, 1] * log2(spec_a[a, 1])
            if peak_b_int>0:
                entropy_b += -peak_b_int * log2(peak_b_int)
            peak_cur = (spec_a[a, 1] + peak_b_int)/2.
            entropy_merged += -peak_cur*log2(peak_cur)

            peak_b_int = 0.
            a += 1
        elif mass_delta_da > ms2_da:
            # Peak only existed in spec b.
            entropy_b += -spec_b[b, 1] * log2(spec_b[b, 1])
            peak_cur = spec_b[b, 1]/2.
            entropy_merged += -peak_cur*log2(peak_cur)

            b += 1
        else:
            # Peak existed in both spec.
            peak_b_int += spec_b[b, 1]
            b += 1

    if peak_b_int > 0.:
        entropy_a += -spec_a[a, 1] * log2(spec_a[a, 1])
        entropy_b += -peak_b_int * log2(peak_b_int)
        peak_cur = (spec_a[a, 1] + peak_b_int)/2.
        entropy_merged += -peak_cur*log2(peak_cur)

        peak_b_int = 0.
        a += 1

    # Fill the rest into merged spec
    for i in range(b,spec_b.shape[0]):
        entropy_b += -spec_b[i, 1] * log2(spec_b[i, 1])
        peak_cur = spec_b[i, 1]/2.
        entropy_merged += -peak_cur*log2(peak_cur)

    for i in range(a,spec_a.shape[0]):
        entropy_a += -spec_a[i, 1] * log2(spec_a[i, 1])
        peak_cur = spec_a[i, 1]/2.
        entropy_merged += -peak_cur*log2(peak_cur)

    #print(entropy_a, entropy_b, entropy_merged)
    return 1 - entropy_merged + 0.5 * (entropy_a + entropy_b)

