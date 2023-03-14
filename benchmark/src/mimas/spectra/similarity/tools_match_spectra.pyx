# cython: infer_types=True

import numpy as np
cimport numpy as np
from libc.math cimport log2,log,pow

ctypedef np.float32_t float32
ctypedef np.int64_t int_64
ctypedef np.int64_t np_int_64

cpdef np.ndarray match_two_spectra_fast(float32[:,::1] spec_a,float32[:,::1] spec_b,float32 ms2_ppm,float32 ms2_da):
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :return: list. Each element in the list is a list contain three elements:
                              m/z, intensity from spec 1; intensity from spec 2.
    """
    cdef int a = 0
    cdef int b = 0
    cdef int i

    cdef float32[:,::1] spec_merged = np.zeros((spec_a.shape[0]+spec_b.shape[0],3),dtype=np.float32)
    cdef int spec_merged_len=0
    cdef float32 peak_b_int = 0.
    cdef float32 mass_delta_da

    with nogil:
        while a < spec_a.shape[0] and b < spec_b.shape[0]:
            if ms2_ppm > 0:
                ms2_da = ms2_ppm * 1e6 * spec_a[a,0]
            mass_delta_da = spec_a[a, 0] - spec_b[b, 0]

            if mass_delta_da < -ms2_da:
                # Peak only existed in spec a.
                spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1], spec_merged[spec_merged_len,2] = \
                    spec_a[a, 0], spec_a[a, 1], peak_b_int
                spec_merged_len += 1
                peak_b_int = 0.
                a += 1
            elif mass_delta_da > ms2_da:
                # Peak only existed in spec b.
                spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1], spec_merged[spec_merged_len,2] = \
                    spec_b[b, 0], 0., spec_b[b, 1]
                spec_merged_len += 1
                peak_b_int = 0.
                b += 1
            else:
                # Peak existed in both spec.
                peak_b_int += spec_b[b, 1]
                b += 1

        if peak_b_int > 0.:
            spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1], spec_merged[spec_merged_len,2] = \
                    spec_a[a, 0], spec_a[a, 1], peak_b_int
            spec_merged_len += 1
            peak_b_int = 0.
            a += 1

        # Fill the rest into merged spec
        for i in range(b, spec_b.shape[0]):
            spec_merged[spec_merged_len,0],spec_merged[spec_merged_len,1],spec_merged[spec_merged_len,2]= \
                spec_b[i,0], 0., spec_b[i,1]
            spec_merged_len+=1

        for i in range(a, spec_a.shape[0]):
            spec_merged[spec_merged_len,0],spec_merged[spec_merged_len,1],spec_merged[spec_merged_len,2]= \
                spec_a[i,0],spec_a[i,1],0
            spec_merged_len+=1

    # Shrink the merged spec.
    if spec_merged_len==0:
        spec_merged_len=1
    spec_merged = spec_merged[:spec_merged_len,:]
    return np.array(spec_merged)


def match_spectrum_output_number(spec_a, spec_b, ms2_ppm=None, ms2_da=None) :
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    """
    if ms2_da is None:
        ms2_da=-1
    else:
        ms2_ppm=-1

    spec_a=np.asarray(spec_a,dtype=np.float32,order="C")
    spec_b=np.asarray(spec_b,dtype=np.float32,order="C")
    return np.asarray(match_peaks_output_number_c(spec_a,spec_b,float(ms2_ppm),float(ms2_da)))

cdef np_int_64[:,:] match_peaks_output_number_c(const float32[:,:] peaks_a,const float32[:,:] peaks_b,float32 ms2_ppm,float32 ms2_da):
    # TODO: Check this!

    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :return: list. Each element in the list is a list contain three elements:
                              m/z, intensity from spec 1; intensity from spec 2.
    """
    cdef int a = 0
    cdef int b = 0
    cdef int i

    cdef np_int_64[:,:] spec_merged = np.zeros((peaks_a.shape[0]+peaks_b.shape[0],2),dtype=np.int64)
    cdef int spec_merged_len=0
    cdef float32 peak_b_int = 0.
    cdef int peak_b_no=-1
    cdef float32 mass_delta_da

    with nogil:
    #if 1:
        while a < peaks_a.shape[0] and b < peaks_b.shape[0]:
            if ms2_ppm > 0:
                ms2_da = ms2_ppm * 1e6 * peaks_a[a,0]
            mass_delta_da = peaks_a[a, 0] - peaks_b[b, 0]

            if mass_delta_da < -ms2_da:
                # Peak only existed in spec a.
                spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = a, peak_b_no
                spec_merged_len += 1
                peak_b_int = 0.
                peak_b_no = -1
                a += 1
            elif mass_delta_da > ms2_da:
                # Peak only existed in spec b.
                b += 1
            else:
                # Peak existed in both spec.
                if peak_b_int > 0:
                    if peak_b_int > peaks_b[b, 1]:
                        # Use previous one
                        spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = a, peak_b_no
                    else:
                        # Use this one
                        spec_merged[spec_merged_len+1,0], spec_merged[spec_merged_len+1,1] = a, b

                    spec_merged_len += 1
                    a += 1
                    peak_b_int = 0.
                    peak_b_no = -1
                else:
                    # Record this one
                    peak_b_int = peaks_b[b, 1]
                    peak_b_no = b
                b += 1

        if peak_b_int > 0.:
            spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = a, peak_b_no
            spec_merged_len += 1
            a += 1

        # Fill the rest into merged spec
        for i in range(a,peaks_a.shape[0]):
            spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = i, -1
            spec_merged_len += 1

    # Shrink the merged spec.
    if spec_merged_len==0:
        spec_merged_len+=1
    spec_merged = spec_merged[:spec_merged_len,:]
    return spec_merged

