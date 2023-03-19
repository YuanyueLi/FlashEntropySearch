import multiprocessing
import pickle
import struct
from pathlib import Path

import numpy as np

try:
    import cupy as cp
    entropy_transform = cp.ElementwiseKernel(
        'T intensity_a, T intensity_b',
        'T similarity',
        '''T intensity_ab = intensity_a + intensity_b;
           similarity = intensity_ab * log2f(intensity_ab) - intensity_a * log2f(intensity_a) - intensity_b * log2f(intensity_b);'''
    )
    entropy_transform(cp.array([0.5], dtype=np.float32), cp.array([0.5], dtype=np.float32))
except:
    print('Cupy is not installed or not correctly setted, please install cupy to use GPU acceleration.')

from mimas.helper.multiplecore import convert_numpy_array_to_shared_memory, MPRunner
from mimas.spectra.spectral_entropy_c.functions import apply_weight_to_intensity


class EntropyHybridSearchCore:
    """
    This class will only do the function of entropy hybrid search.
    This class only calculate the product ions
    """

    def __init__(self, path_array=None, mz_index_step=0.0001, max_ms2_tolerance_in_da=0.024):
        if path_array is None:
            self.use_file_array = False
            self.path_array = None
        else:
            self.use_file_array = True
            self.path_array = Path(str(path_array))
            self.path_array.mkdir(parents=True, exist_ok=True)

        self.index = []
        self.mz_index_step = mz_index_step
        self.total_spectra_num = 0
        self.is_index_in_shared_memory = False
        self.max_ms2_tolerance_in_da = max_ms2_tolerance_in_da

    def _preprocess_peaks(self, peaks):
        """
        Preprocess the peaks before indexing.
        """
        peaks_clean = np.asarray(apply_weight_to_intensity(peaks))
        peaks_clean[:, 1] /= 2
        return peaks_clean

    def _score_peaks(self, intensity_query, intensity_library):
        intensity_ab = intensity_library + intensity_query
        modified_value_product = intensity_ab * np.log2(intensity_ab) \
            - intensity_library * np.log2(intensity_library) - intensity_query * np.log2(intensity_query)
        return modified_value_product

    def _score_peaks_gpu(self, intensity_query, intensity_library):
        return entropy_transform(cp.array(intensity_library), intensity_query)

    def build_index(self, all_precursor_mz_list, all_peaks_list, max_indexed_mz=1500.00005):
        """
        Build the index for the spectra library.
        :param all_peaks_list: list of all peaks in the spectra library. 
                                Each element is a 2-D numpy array of shape (n, 2).
                                The n is the number of peaks in the spectra, n >= 1.
                                The sum of all intensities in the spectra is 1.
        """

        total_spectra_num = len(all_peaks_list)
        total_peak_num = sum([peaks.shape[0] for peaks in all_peaks_list])

        if total_spectra_num == 0 or total_peak_num == 0:
            self.index = []
            return self.index
        # total_spectra_num can not be bigger than 2^32-1 (uint32), total_peak_num can not be bigger than 2^63-1 (int64)
        assert total_spectra_num < 4294967295 and total_peak_num < 9223372036854775807

        peak_idx = 0
        dtype_len = 6
        dtype_peak_data = np.dtype([("mz", np.float32),
                                    ("nl_mz", np.float32),
                                    ("intensity", np.float32),
                                    ("spec_idx", np.uint32),
                                    ("peak_idx", np.uint64)], align=True)
        peak_data = np.zeros(total_peak_num, dtype=dtype_peak_data)
        for idx, peaks in enumerate(all_peaks_list):
            assert peaks.shape[0] >= 1  # Check if the number of peaks is greater than 0.
            assert abs(np.sum(peaks[:, 1])-1) < 1e-4  # Check if the sum of all intensities is 1.
            # assert np.all(peaks[:-1, 0] <= peaks[1:, 0])  # Check if the peaks are sorted.
            # Check if the peaks are separated enough.
            assert peaks.shape[0] <= 1 or np.min(peaks[1:, 0] - peaks[:-1, 0]) > self.max_ms2_tolerance_in_da * 2, \
                f'Error found in spectrum {idx}, the peaks are too close to each other. The peaks are {str(peaks)}'

            # Pre-calculate library peaks entropy
            # peaks_clean = np.asarray(apply_weight_to_intensity(peaks))
            # peaks_clean[:, 1] /= 2
            peaks_clean = self._preprocess_peaks(peaks)

            # Store the peaks
            neutral_loss_mz = all_precursor_mz_list[idx]-peaks_clean[:, 0]
            # Store the peaks
            peak_data.view(np.float32).reshape(-1, dtype_len)[peak_idx:(peak_idx + peaks.shape[0]), 0] = peaks_clean[:, 0]
            peak_data.view(np.float32).reshape(-1, dtype_len)[peak_idx:(peak_idx + peaks.shape[0]), 1] = neutral_loss_mz
            peak_data.view(np.float32).reshape(-1, dtype_len)[peak_idx:(peak_idx + peaks.shape[0]), 2] = peaks_clean[:, 1]
            peak_data.view(np.uint32).reshape(-1, dtype_len)[peak_idx:(peak_idx + peaks.shape[0]), 3] = idx
            peak_idx += peaks.shape[0]

        # Generate the index
        peak_data.sort(order="mz")
        peak_data.view(np.uint64).reshape(-1, dtype_len//2)[:, 2] = np.arange(0, total_peak_num, dtype=np.uint64)
        if self.use_file_array:
            peak_data.view(np.float32).reshape(-1, dtype_len)[:, 0].tofile(self.path_array / "all_peaks_mz.npy")
            peak_data.view(np.float32).reshape(-1, dtype_len)[:, 2].tofile(self.path_array / "all_peaks_intensity.npy")
            peak_data.view(np.uint32).reshape(-1, dtype_len)[:, 3].tofile(self.path_array / "all_peaks_spec_idx.npy")

            all_peaks_mz = np.memmap(self.path_array / 'all_peaks_mz.npy', dtype=np.float32, mode='r', shape=(total_peak_num,))
            all_peaks_intensity = np.memmap(self.path_array / 'all_peaks_intensity.npy', dtype=np.float32, mode='r', shape=(total_peak_num,))
            all_peaks_spec_idx = np.memmap(self.path_array / 'all_peaks_spec_idx.npy', dtype=np.uint32, mode='r', shape=(total_peak_num,))
        else:
            all_peaks_mz = np.copy(peak_data.view(np.float32).reshape(-1, dtype_len)[:, 0])
            all_peaks_intensity = np.copy(peak_data.view(np.float32).reshape(-1, dtype_len)[:, 2])
            all_peaks_spec_idx = np.copy(peak_data.view(np.uint32).reshape(-1, dtype_len)[:, 3])

        # Build the index for all_peaks_mz
        max_mz = min(np.max(all_peaks_mz), max_indexed_mz)
        search_array = np.arange(0., max_mz, self.mz_index_step)
        all_peaks_mz_idx_start = np.searchsorted(all_peaks_mz, search_array, side='left').astype(np.int64)
        if self.use_file_array:
            all_peaks_mz_idx_start.tofile(self.path_array / 'all_peaks_mz_idx_start.npy')
            all_peaks_mz_idx_start = np.memmap(self.path_array / 'all_peaks_mz_idx_start.npy', dtype=np.int64, mode='r', shape=(len(search_array),))

        # Sort by neutral loss
        peak_data.sort(order="nl_mz")
        if self.use_file_array:
            peak_data.view(np.float32).reshape(-1, dtype_len)[:, 1].tofile(self.path_array / "all_nl_mz.npy")
            peak_data.view(np.float32).reshape(-1, dtype_len)[:, 2].tofile(self.path_array / "all_nl_intensity.npy")
            peak_data.view(np.uint32).reshape(-1, dtype_len)[:, 3].tofile(self.path_array / "all_nl_spec_idx.npy")
            peak_data.view(np.uint64).reshape(-1, dtype_len//2)[:, 2].tofile(self.path_array / "all_peaks_idx_for_nl.npy")

            all_nl_mz = np.memmap(self.path_array / 'all_nl_mz.npy', dtype=np.float32, mode='r', shape=(total_peak_num,))
            all_nl_intensity = np.memmap(self.path_array / 'all_nl_intensity.npy', dtype=np.float32, mode='r', shape=(total_peak_num,))
            all_nl_spec_idx = np.memmap(self.path_array / 'all_nl_spec_idx.npy', dtype=np.uint32, mode='r', shape=(total_peak_num,))
            all_peaks_idx_for_nl = np.memmap(self.path_array / 'all_peaks_idx_for_nl.npy', dtype=np.uint64, mode='r', shape=(total_peak_num,))
        else:
            all_nl_mz = np.copy(peak_data.view(np.float32).reshape(-1, dtype_len)[:, 1])
            all_nl_intensity = np.copy(peak_data.view(np.float32).reshape(-1, dtype_len)[:, 2])
            all_nl_spec_idx = np.copy(peak_data.view(np.uint32).reshape(-1, dtype_len)[:, 3])
            all_peaks_idx_for_nl = np.copy(peak_data.view(np.uint64).reshape(-1, dtype_len//2)[:, 2])

        # Build the index for all_nl_mz
        max_mz = min(np.max(all_nl_mz), max_indexed_mz)
        search_array = np.arange(0., max_mz, self.mz_index_step)
        all_nl_mz_idx_start = np.searchsorted(all_nl_mz, search_array, side='left').astype(np.int64)
        if self.use_file_array:
            all_nl_mz_idx_start.tofile(self.path_array / 'all_nl_mz_idx_start.npy')
            all_nl_mz_idx_start = np.memmap(self.path_array / 'all_nl_mz_idx_start.npy', dtype=np.int64, mode='r', shape=(len(search_array),))

        self.index = [all_peaks_mz_idx_start, all_peaks_mz, all_peaks_intensity, all_peaks_spec_idx,
                      all_nl_mz_idx_start, all_nl_mz, all_nl_intensity, all_nl_spec_idx, all_peaks_idx_for_nl]
        self.total_spectra_num = total_spectra_num
        return self.index

    def search(self, *, precursor_mz, peaks, ms2_tolerance_in_da, target="cpu", check_peaks=True, **kwargs):
        """
        search_type is 0: search all spectra.
        search_type is 1: search spectra in the range [search_spectra_idx_min, search_spectra_idx_max).
        search_type is 2: search spectra in the array search_array with entry equals 1, the length of search_array should be equal to the self.total_spectra_num
        """
        if not self.index:
            return np.zeros(0, dtype=np.float32)

        if check_peaks:
            assert ms2_tolerance_in_da <= self.max_ms2_tolerance_in_da
            assert abs(np.sum(peaks[:, 1])-1) < 1e-4  # Check if the sum of all intensities is 1.
            # assert np.all(peaks[:-1, 0] <= peaks[1:, 0])  # Check if the peaks are sorted.
            # Check if the peaks are separated enough.
            assert peaks.shape[0] <= 1 or np.max(peaks[1:, 0] - peaks[:-1, 0]) > self.max_ms2_tolerance_in_da * 2

        library_peaks_mz_idx_start, library_peaks_mz, library_peaks_intensity, library_peaks_spec_idx, \
            library_nl_mz_idx_start, library_nl_mz, library_nl_intensity, library_nl_spec_idx, library_peaks_idx_for_nl = self.index
        index_number_in_one_da = int(1/self.mz_index_step)

        # Calculate query peaks entropy
        # peaks = np.asarray(apply_weight_to_intensity(peaks))
        # peaks[:, 1] /= 2
        peaks = self._preprocess_peaks(peaks)

        # Go through all peak in the spectrum and determine the mz index range
        product_peak_match_idx_min = np.zeros(peaks.shape[0], dtype=np.uint64)
        product_peak_match_idx_max = np.zeros(peaks.shape[0], dtype=np.uint64)
        for peak_idx, (mz, intensity) in enumerate(peaks):
            # Determine the mz index range
            product_mz_idx_min = calculate_index_for_product_ion_mz(
                mz - ms2_tolerance_in_da, library_peaks_mz, library_peaks_mz_idx_start, 'left', index_number_in_one_da)
            product_mz_idx_max = calculate_index_for_product_ion_mz(
                mz + ms2_tolerance_in_da, library_peaks_mz, library_peaks_mz_idx_start, 'right', index_number_in_one_da)

            product_peak_match_idx_min[peak_idx] = product_mz_idx_min
            product_peak_match_idx_max[peak_idx] = product_mz_idx_max

        # duplicated_array = []
        if target == "cpu":
            entropy_similarity = np.zeros(self.total_spectra_num, dtype=np.float32)

            # Go through all the peaks in the spectrum and calculate the entropy similarity
            for peak_idx, (mz, intensity) in enumerate(peaks):
                ###############################################################
                # Match the original product ion
                product_mz_idx_min = product_peak_match_idx_min[peak_idx]
                product_mz_idx_max = product_peak_match_idx_max[peak_idx]

                # Calculate the entropy similarity for this matched peak
                # intensity_library = library_peaks_intensity[product_mz_idx_min: product_mz_idx_max]
                # intensity_ab = intensity_library + intensity
                # modified_value_product = intensity_ab * np.log2(intensity_ab) \
                # - intensity_library * np.log2(intensity_library) - intensity * np.log2(intensity)
                modified_idx_product = library_peaks_spec_idx[product_mz_idx_min: product_mz_idx_max]
                modified_value_product = self._score_peaks(intensity, library_peaks_intensity[product_mz_idx_min: product_mz_idx_max])

                entropy_similarity[modified_idx_product] += modified_value_product

                ###############################################################
                # Match the neutral loss ions
                mz_nl = precursor_mz-mz
                # Determine the mz index range
                neutral_loss_mz_idx_min = calculate_index_for_product_ion_mz(
                    mz_nl - ms2_tolerance_in_da, library_nl_mz, library_nl_mz_idx_start, 'left', index_number_in_one_da)
                neutral_loss_mz_idx_max = calculate_index_for_product_ion_mz(
                    mz_nl + ms2_tolerance_in_da, library_nl_mz, library_nl_mz_idx_start, 'right', index_number_in_one_da)

                # Calculate the entropy similarity for this matched peak
                # intensity_library = library_nl_intensity[neutral_loss_mz_idx_min: neutral_loss_mz_idx_max]
                # intensity_ab = intensity_library + intensity
                # modified_value_nl = intensity_ab * np.log2(intensity_ab) \
                #     - intensity_library * np.log2(intensity_library) - intensity * np.log2(intensity)
                modified_idx_nl = library_nl_spec_idx[neutral_loss_mz_idx_min: neutral_loss_mz_idx_max]
                modified_value_nl = self._score_peaks(intensity, library_nl_intensity[neutral_loss_mz_idx_min: neutral_loss_mz_idx_max])

                # Check if the neutral loss ion is already matched to other query peak as a product ion
                nl_matched_product_ion_idx = library_peaks_idx_for_nl[neutral_loss_mz_idx_min: neutral_loss_mz_idx_max]
                s1 = np.searchsorted(product_peak_match_idx_min, nl_matched_product_ion_idx, side='right')
                s2 = np.searchsorted(product_peak_match_idx_max-1, nl_matched_product_ion_idx, side='left')
                modified_value_nl[s1 > s2] = 0

                # Check if this query peak is already matched to a product ion in the same library spectrum
                _, _, duplicate_idx_in_nl = np.intersect1d(modified_idx_product, modified_idx_nl, assume_unique=True, return_indices=True)
                modified_value_nl[duplicate_idx_in_nl] = 0

                entropy_similarity[modified_idx_nl] += modified_value_nl

                # duplicated_array.append(modified_idx_nl[s1>s2])
                # duplicated_array.append(modified_idx_nl[duplicate_idx_in_nl])

            return entropy_similarity  # , np.unique(np.concatenate(duplicated_array))

        elif target == "gpu":
            entropy_similarity_array = []
            product_peak_match_idx_min_gpu = cp.array(product_peak_match_idx_min)
            product_peak_match_idx_max_gpu = cp.array(product_peak_match_idx_max-1)

            # Go through all the peaks in the spectrum and calculate the entropy similarity
            for peak_idx, (mz, intensity) in enumerate(peaks):
                ###############################################################
                # Match the original product ion
                product_mz_idx_min = product_peak_match_idx_min[peak_idx]
                product_mz_idx_max = product_peak_match_idx_max[peak_idx]

                print(peak_idx, ":", product_mz_idx_max-product_mz_idx_min)
                # Calculate the entropy similarity for this matched peak
                modified_idx_product = library_peaks_spec_idx[product_mz_idx_min: product_mz_idx_max]
                modified_value_product = self._score_peaks_gpu(intensity, library_peaks_intensity[product_mz_idx_min: product_mz_idx_max])

                entropy_similarity_array.append((modified_idx_product, modified_value_product.get()))
                del modified_value_product

                ###############################################################
                # Match the neutral loss ions
                mz_nl = precursor_mz-mz
                # Determine the mz index range
                neutral_loss_mz_idx_min = calculate_index_for_product_ion_mz(
                    mz_nl - ms2_tolerance_in_da, library_nl_mz, library_nl_mz_idx_start, 'left', index_number_in_one_da)
                neutral_loss_mz_idx_max = calculate_index_for_product_ion_mz(
                    mz_nl + ms2_tolerance_in_da, library_nl_mz, library_nl_mz_idx_start, 'right', index_number_in_one_da)

                print(peak_idx, ":", neutral_loss_mz_idx_max-neutral_loss_mz_idx_min)
                # Calculate the entropy similarity for this matched peak
                modified_idx_nl = library_nl_spec_idx[neutral_loss_mz_idx_min: neutral_loss_mz_idx_max]
                modified_value_nl = self._score_peaks_gpu(intensity, library_nl_intensity[neutral_loss_mz_idx_min: neutral_loss_mz_idx_max])

                # Check if the neutral loss ion is already matched to other query peak as a product ion
                nl_matched_product_ion_idx = cp.array(library_peaks_idx_for_nl[neutral_loss_mz_idx_min: neutral_loss_mz_idx_max])
                s1 = cp.searchsorted(product_peak_match_idx_min_gpu, nl_matched_product_ion_idx, side='right')
                s2 = cp.searchsorted(product_peak_match_idx_max_gpu, nl_matched_product_ion_idx, side='left')
                modified_value_nl[s1 > s2] = 0

                # Check if this query peak is already matched to a product ion in the same library spectrum
                # _, _, duplicate_idx_in_nl = np.intersect1d(modified_idx_product, modified_idx_nl, assume_unique=True, return_indices=True)
                duplicate_idx_in_nl = intersect1d_gpu(modified_idx_product, modified_idx_nl)
                modified_value_nl[duplicate_idx_in_nl] = 0

                entropy_similarity_array.append((modified_idx_nl, modified_value_nl.get()))
                del modified_value_nl

            entropy_similarity = cp.zeros(self.total_spectra_num, dtype=np.float32)
            for idx, value in entropy_similarity_array:
                entropy_similarity.scatter_add(cp.array(idx), cp.array(value))
            return entropy_similarity.get()
        else:
            raise ValueError("target should be cpu or gpu")

    def write_to_file(self, fo):
        information = {
            "mz_index_step": self.mz_index_step,
            "total_spectra_num": self.total_spectra_num,
            "max_ms2_tolerance_in_da": self.max_ms2_tolerance_in_da,
        }
        information_data = pickle.dumps(information)
        if self.use_file_array:
            for array in self.index:
                array.flush()
            with open(self.path_array/"information.pkl", "wb") as f:
                f.write(information_data)
        else:
            for array in self.index:
                write_numpy_array_to_file_stream(fo, array)
            fo.write(struct.pack("i", len(information_data)))
            fo.write(information_data)

    def read_from_file(self, fi=None, use_memmap=False):
        if self.use_file_array:
            if use_memmap:
                self.index = [
                    np.memmap(self.path_array / 'all_peaks_mz_idx_start.npy', dtype=np.int64, mode='r'),
                    np.memmap(self.path_array / 'all_peaks_mz.npy', dtype=np.float32, mode='r'),
                    np.memmap(self.path_array / 'all_peaks_intensity.npy', dtype=np.float32, mode='r'),
                    np.memmap(self.path_array / 'all_peaks_spec_idx.npy', dtype=np.uint32, mode='r'),
                    np.memmap(self.path_array / 'all_nl_mz_idx_start.npy', dtype=np.int64, mode='r'),
                    np.memmap(self.path_array / 'all_nl_mz.npy', dtype=np.float32, mode='r'),
                    np.memmap(self.path_array / 'all_nl_intensity.npy', dtype=np.float32, mode='r'),
                    np.memmap(self.path_array / 'all_nl_spec_idx.npy', dtype=np.uint32, mode='r'),
                    np.memmap(self.path_array / 'all_peaks_idx_for_nl.npy', dtype=np.int64, mode='r'),
                ]
            else:
                self.index = [
                    np.fromfile(self.path_array / 'all_peaks_mz_idx_start.npy', dtype=np.int64),
                    np.fromfile(self.path_array / 'all_peaks_mz.npy', dtype=np.float32),
                    np.fromfile(self.path_array / 'all_peaks_intensity.npy', dtype=np.float32),
                    np.fromfile(self.path_array / 'all_peaks_spec_idx.npy', dtype=np.uint32),
                    np.fromfile(self.path_array / 'all_nl_mz_idx_start.npy', dtype=np.int64),
                    np.fromfile(self.path_array / 'all_nl_mz.npy', dtype=np.float32),
                    np.fromfile(self.path_array / 'all_nl_intensity.npy', dtype=np.float32),
                    np.fromfile(self.path_array / 'all_nl_spec_idx.npy', dtype=np.uint32),
                    np.fromfile(self.path_array / 'all_peaks_idx_for_nl.npy', dtype=np.uint64),
                ]
            information = pickle.load(open(self.path_array/"information.pkl", "rb"))
        else:
            self.index = []
            for _ in range(4):
                self.index.append(read_numpy_array_from_file_stream(fi, use_memmap))
            information_data_length = struct.unpack("i", fi.read(4))[0]
            information = pickle.loads(fi.read(information_data_length))

        self.mz_index_step = information["mz_index_step"]
        self.total_spectra_num = information["total_spectra_num"]
        self.max_ms2_tolerance_in_da = information["max_ms2_tolerance_in_da"]

        if self.total_spectra_num == 0:
            if "precursor_mz" in vars(self):
                self.total_spectra_num = len(self.precursor_mz)
            else:
                self.total_spectra_num = np.max(self.index[-1])+1

    def move_index_array_to_shared_memory(self):
        if self.is_index_in_shared_memory:
            return

        for i, array in enumerate(self.index):
            self.index[i] = convert_numpy_array_to_shared_memory(array)
        self.is_index_in_shared_memory = True


def calculate_index_for_product_ion_mz(product_mz, library_peaks_mz, library_peaks_mz_idx_start, side, index_number):
    product_mz_min_int = (np.floor(product_mz*index_number)).astype(int)
    product_mz_max_int = product_mz_min_int + 1

    if product_mz_min_int >= len(library_peaks_mz_idx_start):
        product_mz_idx_search_start = library_peaks_mz_idx_start[-1]
    else:
        product_mz_idx_search_start = library_peaks_mz_idx_start[product_mz_min_int].astype(int)

    if product_mz_max_int >= len(library_peaks_mz_idx_start):
        product_mz_idx_search_end = len(library_peaks_mz)
    else:
        product_mz_idx_search_end = library_peaks_mz_idx_start[product_mz_max_int].astype(int)+1

    return product_mz_idx_search_start + np.searchsorted(
        library_peaks_mz[product_mz_idx_search_start:product_mz_idx_search_end], product_mz, side=side)


def write_numpy_array_to_file_stream(file_stream, array):
    array_bytes = array.tobytes(order='C')
    array_info = array.dtype, array.shape
    array_info_bytes = pickle.dumps(array_info)
    file_stream.write(struct.pack('QQ', len(array_info_bytes), len(array_bytes)))
    file_stream.write(array_info_bytes)
    file_stream.write(array_bytes)


def read_numpy_array_from_file_stream(file_stream, use_memmap=False):
    array_info_bytes = file_stream.read(struct.calcsize('QQ'))
    array_info_length, array_length = struct.unpack('QQ', array_info_bytes)
    array_info_bytes = file_stream.read(array_info_length)
    array_dtype, array_shape = pickle.loads(array_info_bytes)
    if use_memmap:
        location = file_stream.tell()
        array = np.memmap(file_stream.name, dtype=array_dtype, mode='r', offset=location, shape=array_shape)
        file_stream.seek(location + array_length)
    else:
        array_bytes = file_stream.read(array_length)
        array = np.frombuffer(array_bytes, dtype=array_dtype).reshape(array_shape)
    return array


class CosineHybridSearchCore(EntropyHybridSearchCore):
    def __init__(self, intensity_power=1., path_array=None, mz_index_step=0.0001, max_ms2_tolerance_in_da=0.024):
        self.intensity_power = intensity_power
        super().__init__(path_array, mz_index_step, max_ms2_tolerance_in_da)

    def _preprocess_peaks(self, peaks):
        peaks_clean = np.copy(np.asarray(peaks))
        peaks_clean[:, 1] = np.power(peaks_clean[:, 1], self.intensity_power)
        intensity_sum = np.sqrt(np.sum(np.power(peaks_clean[:, 1], 2)))
        peaks_clean[:, 1] = peaks_clean[:, 1] / intensity_sum
        return peaks_clean

    def _score_peaks(self, intensity_query, intensity_library):
        return intensity_query*intensity_library

    def _score_peaks_gpu(self, intensity_query, intensity_library):
        return cp.array(intensity_library)*intensity_query


def intersect1d_gpu(ar1, ar2):
    aux = cp.array(np.concatenate((ar1, ar2)))
    # aux = cp.concatenate((cp.array(ar1), cp.array(ar2)))
    aux_sort_indices = cp.argsort(aux)
    aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]
    ar2_indices = aux_sort_indices[1:][mask] - ar1.size
    return ar2_indices


def intersect1d_cpu(ar1, ar2):
    aux = np.concatenate((ar1, ar2))
    aux_sort_indices = np.argsort(aux)
    aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]
    # int1d = aux[:-1][mask]

    # ar1_indices = aux_sort_indices[:-1][mask]
    ar2_indices = aux_sort_indices[1:][mask] - ar1.size
    return ar2_indices
