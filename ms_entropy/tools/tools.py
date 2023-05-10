#!/usr/bin/env python3
import numpy as np
from .fast_spectrum import centroid_spectrum


def clean_spectrum(spectrum: np.ndarray,
                   min_mz: float = None, max_mz: float = None,
                   noise_threshold=0.01,
                   min_ms2_difference_in_da: float = 0.05,
                   max_peak_num: int = None,
                   normalize_intensity: bool = True) -> np.ndarray:
    """
    Clean the spectrum with the following steps:

    0. The empty peaks (m/z = 0 or intensity = 0) will be removed.

    1. Remove the peaks have m/z lower than the min_mz.

    2. Remove the peaks have m/z higher than the max_mz. This step can be used for remove precursor ions.

    3. Centroid the spectrum by merging the peaks within the +/- min_ms2_difference_in_da, sort the result spectrum by m/z.

    4. Remove the peaks with intensity less than the noise_threshold * maximum (intensity).

    5. Keep the top max_peak_num peaks, and remove the rest peaks.
    
    6. Normalize the intensity to sum to 1.

    :param spectrum: The spectrum, a 2D numpy array with shape (n, 2), the first column is m/z, the second column is intensity.
    :param min_mz: The minimum m/z to keep, if None, all the peaks will be kept. Default is None.
    :param max_mz: The maximum m/z to keep, if None, all the peaks will be kept. Default is None.
    :param noise_threshold: The noise threshold, peaks have intensity lower than
                            noise_threshold * maximum (intensity) will be removed.
                            If None, all the peaks will be kept.
                            Default is 0.01.
    :param min_ms2_difference_in_da: The minimum m/z difference to merge the peaks.
                               Default is 0.05.
    :param max_peak_num: The maximum number of peaks to keep. If None, all the peaks will be kept.
                            Default is None.
    :param normalize_intensity: Whether to normalize the intensity to sum to 1.
                                Default is True.

    :return: The cleaned spectrum, a 2D numpy array with shape (n, 2), the first column is m/z, the second column is intensity.
    """

    # Check the input spectrum and convert it to numpy array with shape (n, 2) and dtype np.float32.
    spectrum = np.asarray(spectrum, dtype=np.float32, order="C")
    if len(spectrum) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    assert spectrum.ndim == 2 and spectrum.shape[1] == 2, "The input spectrum must be a 2D numpy array with shape (n, 2)."

    # Step 0: Remove the empty peaks (m/z = 0 or intensity = 0).
    spectrum = spectrum[np.bitwise_and(spectrum[:, 0] > 0, spectrum[:, 1] > 0)]

    # Step 1: Remove the peaks have m/z lower than the min_mz.
    if min_mz is not None:
        spectrum = spectrum[spectrum[:, 0] >= min_mz]

    # Step 2: Remove the peaks have m/z higher than the max_mz. This step can be used for remove precursor ions.
    if max_mz is not None:
        spectrum = spectrum[spectrum[:, 0] <= max_mz]

    # Step 3: Centroid the spectrum by merging the peaks within the +/- min_ms2_difference, sort the result spectrum by m/z.
    # Sort the spectrum by m/z.
    spectrum = spectrum[np.argsort(spectrum[:, 0])]
    # Centroid the spectrum by merging the peaks within the +/- min_ms2_difference.
    spectrum = centroid_spectrum(spectrum, ms2_ppm=None, ms2_da=min_ms2_difference_in_da)

    if spectrum.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # Step 4: Remove the peaks with intensity less than the noise_threshold * maximum (intensity).
    if noise_threshold is not None:
        spectrum = spectrum[spectrum[:, 1] >= noise_threshold * np.max(spectrum[:, 1])]

    # Step 5: Keep the top max_peak_num peaks, and remove the rest peaks.
    if max_peak_num is not None:
        # Sort the spectrum by intensity.
        spectrum = spectrum[np.argsort(spectrum[:, 1])[-max_peak_num:]]
        # Sort the spectrum by m/z.
        spectrum = spectrum[np.argsort(spectrum[:, 0])]

    # Step 6: Normalize the intensity to sum to 1.
    if normalize_intensity:
        spectrum_sum = np.sum(spectrum[:, 1])
        if spectrum_sum > 0:
            spectrum[:, 1] /= spectrum_sum
        else:
            spectrum = np.zeros((0, 2), dtype=np.float32)

    return spectrum
