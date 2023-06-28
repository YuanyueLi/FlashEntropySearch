import numpy as np
from typing import Union
from mimas.chem.isotope import isotope_distribution
from .tools_spectrum import centroid_spectrum
from .tools_match_spectra import match_two_spectra_fast


def clean_spectrum(spectrum: np.ndarray,
                   min_mz: float = None, max_mz: float = None,
                   noise_threshold=0.01,
                   max_peak_num: int = None,
                   remove_isotope: bool = True,
                   normalize_intensity: bool = True,
                   ms2_ppm: float = None, ms2_da: float = 0.05) -> np.ndarray:
    """
    Clean the spectrum with the following steps:
    1. Remove the peaks have m/z higher than the max_mz. This step can be used for
       remove precursor ions.
    2. Centroid the spectrum, merge the peaks within the +/- ms2_ppm or +/- ms2_da, sort the result spectrum by m/z.
    3. Remove the peaks with intensity less than the noise_threshold * maximum (intensity).
    4. Normalize the intensity to sum to 1.

    At least one of the ms2_ppm or ms2_da need be not None. If both ms2_da and ms2_ppm is given, ms2_da will be used.

    :param spectrum: The spectrum.
    :param max_mz: The maximum m/z to keep, if None, all the peaks will be kept.
    :param noise_threshold: The noise threshold, peaks have intensity lower than
                            noise_threshold * maximum (intensity) will be removed.
                            If None, all the peaks will be kept.
    :param max_peak_num: The maximum number of peaks to keep. If None, all the peaks will be kept.
    :param ms2_ppm: The mass accuracy in ppm.
    :param ms2_da: The mass accuracy in Da.
    :return: The cleaned spectrum.
    """
    # Check the input.
    if ms2_ppm is None and ms2_da is None:
        raise RuntimeError("Either ms2_ppm or ms2_da should be given!")

    # Convert the spectrum to numpy array.
    spectrum = convert_spectrum_to_numpy_array(spectrum)

    # Remove the empty peaks.
    spectrum = spectrum[np.bitwise_and(spectrum[:, 0] > 0, spectrum[:, 1] > 0)]

    # 1. Remove the peaks have m/z higher than the max_mz.
    if min_mz is not None:
        spectrum = spectrum[spectrum[:, 0] >= min_mz]
    if max_mz is not None:
        spectrum = spectrum[spectrum[:, 0] <= max_mz]

    # Sort spectrum by m/z.
    spectrum = spectrum[np.argsort(spectrum[:, 0])]
    # 2. Centroid the spectrum, merge the peaks within the +/- ms2_ppm or +/- ms2_da.
    spectrum = centroid_spectrum(spectrum, ms2_ppm=ms2_ppm, ms2_da=ms2_da)

    # 3. Remove the isotope peaks.
    if remove_isotope:
        spectrum = remove_isotope_peak(spectrum, ms2_da=ms2_da, ms2_ppm=ms2_ppm)

    # 4. Remove the peaks with intensity less than the noise_threshold * maximum (intensity).
    if noise_threshold is not None and spectrum.shape[0] > 0:
        spectrum = spectrum[spectrum[:, 1] >= noise_threshold * np.max(spectrum[:, 1])]

    # 5. Select the top max_peak_num peaks.
    if max_peak_num is not None and spectrum.shape[0] > 0:
        spectrum = spectrum[np.argsort(spectrum[:, 1])[-max_peak_num:]]
        spectrum = spectrum[np.argsort(spectrum[:, 0])]

    # 6. Normalize the intensity to sum to 1.
    if normalize_intensity:
        spectrum_sum = np.sum(spectrum[:, 1])
        if spectrum_sum > 0:
            spectrum[:, 1] /= spectrum_sum
            return spectrum

    return spectrum


def remove_isotope_peak(spectrum: np.ndarray, ms2_da: float = 0.05, ms2_ppm: float = None) -> np.ndarray:
    """
    Remove the isotope peaks.
    The intensity of the isotope peaks will be added to the intensity of the parent peak,
    then, the isotope peaks will be removed.
    This function will modify the input spectrum, but it will not change the total intensity of the original spectrum.
    """
    mass_neutron = 1.00866491588
    is_peak_isotope = np.zeros(spectrum.shape[0], dtype=np.int32)
    # Label the isotope groups.
    for i, peak in enumerate(spectrum):
        if is_peak_isotope[i] == 1:
            continue
        # Look for the isotope peak
        if ms2_da is None:
            ms2_da_cur = ms2_ppm * peak[0] / 1e6
        else:
            ms2_da_cur = ms2_da

        base_mz, base_intensity = peak[0], peak[1]
        base_peak_id = i

        next_mz = base_mz + mass_neutron
        base_mz_int = int(base_mz)
        if base_mz_int >= len(isotope_distribution):
            base_mz_int = len(isotope_distribution)-1
        isotope_distribution_cur = isotope_distribution[base_mz_int]
        isotope_id = 0
        while isotope_id < len(isotope_distribution_cur):
            i += 1
            if i >= spectrum.shape[0] or spectrum[i, 0] > next_mz+ms2_da_cur:
                break
            elif is_peak_isotope[i] == 1 or spectrum[i, 0] < next_mz-ms2_da_cur:
                continue
            # Look up the isotope peak table to check if the peak is an isotope peak.
            if spectrum[i, 1] > base_intensity * isotope_distribution_cur[isotope_id]:
                break

            # This is an isotope peak, merge its intensity to the base peak.
            spectrum[base_peak_id, 1] += spectrum[i, 1]
            spectrum[i, 1] = 0
            is_peak_isotope[i] = 1

            # Get the next isotope peak.
            isotope_id += 1
            next_mz += mass_neutron

    # Remove the zero intensity peaks.
    spectrum = spectrum[spectrum[:, 1] > 0]
    return spectrum


def convert_spectrum_to_numpy_array(spectrum: Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert the spectrum to numpy array.
    """
    spectrum = np.asarray(spectrum, dtype=np.float32, order="C")
    if spectrum.shape[0] == 0:
        return np.zeros(0, dtype=np.float32, order="C").reshape(-1, 2)
    if spectrum.ndim != 2:
        raise RuntimeError("Error in input spectrum format!")
    return spectrum


def match_two_spectra(spec_a: np.ndarray, spec_b: np.ndarray,
                      ms2_ppm: float = None, ms2_da: float = 0.05) -> np.ndarray:
    """
    - Match two spectra's peak, if the m/z difference of two peaks is within ms2_da or ms2_ppm, 
    the two peaks will be considered as the matched peak.

    - The input spectra should be in the standard numpy array format, the function will not check the format. The format should meet:
        1. The first column is m/z, the second column is intensity.
        2. The spectra should be sorted by m/z.

    :param spec_a: The first spectrum, the dimension should be n,2
    :param spec_b: The second spectrum, the dimension should be m,2
    :param ms2_ppm: The mass accuracy in ppm.
    :param ms2_da: The mass accuracy in Da.
    :return: The matched spectrum, the dimension will be x,3.
    The [x,0] will be the m/z of spec_a or spec_b the [x,1] will be the intensity of spec_a, the [x,2] will be the intensity of spec_b
    """
    spec_a = np.asarray(spec_a, dtype=np.float32, order="C")
    spec_b = np.asarray(spec_b, dtype=np.float32, order="C")
    if ms2_ppm is None:
        return match_two_spectra_fast(spec_a, spec_b, ms2_ppm=-1, ms2_da=ms2_da)
    else:
        return match_two_spectra_fast(spec_a, spec_b, ms2_ppm=ms2_ppm, ms2_da=-1)
