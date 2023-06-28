# cython: infer_types=True

import numpy as np
cimport numpy as np
from libc.math cimport log2,log,pow

ctypedef np.float32_t float32
ctypedef np.int64_t int_64




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



cdef int check_centroid_c(float32[:,::1] spectrum,double ms2_ppm, double ms2_da) nogil:
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


#################################### TO Check ###############################################
################################
# Sort spectra
cdef void sort_spectrum_by_mz_in_place_c(float32[:,:] spec) nogil:
    cdef long n = spec.shape[0]
    cdef long new_n = 0
    cdef long i

    while n > 1:
        new_n = 0
        for i in range(1, n):
            if spec[i - 1][0] > spec[i][0]:
                spec[i - 1][0], spec[i][0] = spec[i][0], spec[i - 1][0]
                spec[i - 1][1], spec[i][1] = spec[i][1], spec[i - 1][1]
                new_n = i
        n = new_n


cdef void sort_spectrum_by_intensity_in_place_c(float32[:,:] spec) nogil:
    cdef long n = spec.shape[0]
    cdef long new_n = 0
    cdef long i

    while n > 1:
        new_n = 0
        for i in range(1, n):
            if spec[i - 1][1] > spec[i][1]:
                spec[i - 1][0], spec[i][0] = spec[i][0], spec[i - 1][0]
                spec[i - 1][1], spec[i][1] = spec[i][1], spec[i - 1][1]
                new_n = i
        n = new_n
################################
