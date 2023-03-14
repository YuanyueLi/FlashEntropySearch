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

from .multiplecore import convert_numpy_array_to_shared_memory, MPRunner
from .entropy_search_core_fast import entropy_similarity_search_fast
from .functions import apply_weight_to_intensity


class EntropySearchCore:
    """
    This class will only do the core function of entropy search.
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

    def build_index(self, all_peaks_list, max_indexed_mz=1500.00005):
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

        if self.use_file_array:
            all_peaks_mz = np.memmap(self.path_array / 'all_peaks_mz.npy', dtype=np.float32, mode='w+', shape=(total_peak_num,))
            all_peaks_intensity = np.memmap(self.path_array / 'all_peaks_intensity.npy', dtype=np.float32, mode='w+', shape=(total_peak_num,))
            all_peaks_spec_idx = np.memmap(self.path_array / 'all_peaks_spec_idx.npy', dtype=np.uint32, mode='w+', shape=(total_peak_num,))
        else:
            all_peaks_mz = np.zeros(total_peak_num, dtype=np.float32)
            all_peaks_intensity = np.zeros(total_peak_num, dtype=np.float32)
            all_peaks_spec_idx = np.zeros(total_peak_num, dtype=np.uint32)

        peak_idx = 0
        for idx, peaks in enumerate(all_peaks_list):
            assert peaks.shape[0] >= 1  # Check if the number of peaks is greater than 0.
            assert abs(np.sum(peaks[:, 1])-1) < 1e-4  # Check if the sum of all intensities is 1.
            # assert np.all(peaks[:-1, 0] <= peaks[1:, 0])  # Check if the peaks are sorted.
            # Check if the peaks are separated enough.
            assert peaks.shape[0] <= 1 or np.min(peaks[1:, 0] - peaks[:-1, 0]) > self.max_ms2_tolerance_in_da * 2, \
                f'Error found in spectrum {idx}, the peaks are too close to each other. The peaks are {str(peaks)}'

            # Pre-calculate library peaks entropy
            peaks_clean = np.asarray(apply_weight_to_intensity(peaks))
            peaks_clean[:, 1] /= 2

            # Store the peaks
            all_peaks_mz[peak_idx:(peak_idx + peaks.shape[0])] = peaks_clean[:, 0]
            all_peaks_intensity[peak_idx:(peak_idx + peaks.shape[0])] = peaks_clean[:, 1]
            all_peaks_spec_idx[peak_idx:(peak_idx + peaks.shape[0])] = idx
            peak_idx += peaks.shape[0]

        # Generate the index
        idx_argsort = np.argsort(all_peaks_mz)
        all_peaks_mz[:] = all_peaks_mz[idx_argsort]
        all_peaks_intensity[:] = all_peaks_intensity[idx_argsort]
        all_peaks_spec_idx[:] = all_peaks_spec_idx[idx_argsort]

        max_mz = min(np.max(all_peaks_mz), max_indexed_mz)
        search_array = np.arange(0., max_mz, self.mz_index_step)
        all_peaks_mz_idx_start = np.searchsorted(all_peaks_mz, search_array, side='left').astype(np.int64)
        if self.use_file_array:
            all_peaks_mz_idx_start_mem = np.memmap(
                self.path_array / 'all_peaks_mz_idx_start.npy', dtype=np.int64, mode='w+', shape=(all_peaks_mz_idx_start.shape[0],))
            all_peaks_mz_idx_start_mem[:] = all_peaks_mz_idx_start
            all_peaks_mz_idx_start = all_peaks_mz_idx_start_mem

        self.index = [all_peaks_mz_idx_start, all_peaks_mz, all_peaks_intensity, all_peaks_spec_idx]
        self.total_spectra_num = total_spectra_num
        return self.index

    def search(self, *, peaks, ms2_tolerance_in_da,
               search_type=0, search_spectra_idx_min=0, search_spectra_idx_max=0, search_array=None,
               target="cpu", check_peaks=True, **kwargs):
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
            assert np.all(peaks[:-1, 0] <= peaks[1:, 0])  # Check if the peaks are sorted.
            # Check if the peaks are separated enough.
            assert peaks.shape[0] <= 1 or np.max(peaks[1:, 0] - peaks[:-1, 0]) > self.max_ms2_tolerance_in_da * 2

        library_peaks_mz_idx_start, library_peaks_mz, library_peaks_intensity, library_peaks_spec_idx = self.index
        index_number_in_one_da = int(1/self.mz_index_step)

        # Calculate query peaks entropy
        peaks = np.asarray(apply_weight_to_intensity(peaks))
        peaks[:, 1] /= 2

        if target == "cpu":
            entropy_similarity = np.zeros(self.total_spectra_num, dtype=np.float32)
            # Go through all the peaks in the spectrum
            for mz, intensity_half in peaks:
                # Determine the mz index range
                product_mz_idx_min = calculate_index_for_product_ion_mz(
                    mz - ms2_tolerance_in_da, library_peaks_mz, library_peaks_mz_idx_start, 'left', index_number_in_one_da)
                product_mz_idx_max = calculate_index_for_product_ion_mz(
                    mz + ms2_tolerance_in_da, library_peaks_mz, library_peaks_mz_idx_start, 'right', index_number_in_one_da)

                if search_type == 0:
                    intensity_library = library_peaks_intensity[product_mz_idx_min: product_mz_idx_max]
                    intensity_ab = intensity_library + intensity_half

                    modified_value = \
                        intensity_ab * np.log2(intensity_ab) - \
                        intensity_library * np.log2(intensity_library) - \
                        intensity_half * np.log2(intensity_half)
                    modified_idx = library_peaks_spec_idx[product_mz_idx_min: product_mz_idx_max]

                    entropy_similarity[modified_idx] += modified_value
                else:
                    entropy_similarity_search_fast(product_mz_idx_min, product_mz_idx_max, intensity_half, entropy_similarity,
                                                   library_peaks_intensity, library_peaks_spec_idx,
                                                   search_type, search_spectra_idx_min, search_spectra_idx_max, search_array)
            return entropy_similarity
        elif target == "gpu":
            entropy_similarity = cp.zeros(self.total_spectra_num, dtype=np.float32)
            for mz, intensity_half in peaks:
                # Determine the mz index range
                product_mz_idx_min = calculate_index_for_product_ion_mz(
                    mz - ms2_tolerance_in_da, library_peaks_mz, library_peaks_mz_idx_start, 'left', index_number_in_one_da)
                product_mz_idx_max = calculate_index_for_product_ion_mz(
                    mz + ms2_tolerance_in_da, library_peaks_mz, library_peaks_mz_idx_start, 'right', index_number_in_one_da)

                intensity_library = cp.array(library_peaks_intensity[product_mz_idx_min: product_mz_idx_max])
                modified_value = entropy_transform(intensity_library, intensity_half)
                modified_idx = cp.array(library_peaks_spec_idx[product_mz_idx_min: product_mz_idx_max])
                entropy_similarity.scatter_add(modified_idx, modified_value)

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

    def read_from_file(self, fi, use_memmap=False):
        if self.use_file_array:
            if use_memmap:
                self.index = [
                    np.memmap(self.path_array / 'all_peaks_mz_idx_start.npy', dtype=np.int64, mode='r'),
                    np.memmap(self.path_array / 'all_peaks_mz.npy', dtype=np.float32, mode='r'),
                    np.memmap(self.path_array / 'all_peaks_intensity.npy', dtype=np.float32, mode='r'),
                    np.memmap(self.path_array / 'all_peaks_spec_idx.npy', dtype=np.uint32, mode='r'),
                ]
            else:
                self.index = [
                    np.fromfile(self.path_array / 'all_peaks_mz_idx_start.npy', dtype=np.int64),
                    np.fromfile(self.path_array / 'all_peaks_mz.npy', dtype=np.float32),
                    np.fromfile(self.path_array / 'all_peaks_intensity.npy', dtype=np.float32),
                    np.fromfile(self.path_array / 'all_peaks_spec_idx.npy', dtype=np.uint32)
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


class EntropySearch(EntropySearchCore):
    """
    This class consider the precursor ion and product ions
    """

    def __init__(self, path_array=None, **kwargs):
        """
        :param path_array: the path to the array files, if None, the array will be stored in the memory
        """
        super().__init__(path_array=path_array, **kwargs)
        self.precursor_mz: np.ndarray = np.zeros(0, dtype=np.float32)

    def build_index(self, all_spectra, sort_by_precursor_mz=True):
        """
        This function will build the index from the spectra generator.
        The spectra need to be sorted by precursor m/z.
        The peaks need to be cleaned and normalize to 1 before using this function.
        This function will use the input as is, and will not do any pre-processing except the intensity weighting by spectral entropy.
        The following keys are used:
            "precursor_mz": precursor m/z
            "peaks": a numpy array of the peaks, with the first column as the m/z and the second column as the intensity

        """
        if sort_by_precursor_mz:
            all_spectra_ori = all_spectra
            all_spectra = [x for x in all_spectra_ori]
            all_spectra.sort(key=lambda x: x["precursor_mz"])

        self.precursor_mz = np.array([x["precursor_mz"] for x in all_spectra], dtype=np.float32)
        if self.use_file_array:
            precursor_mz_file = np.memmap(filename=self.path_array/"precursor_mz.npy",
                                          shape=self.precursor_mz.shape, dtype=self.precursor_mz.dtype, mode="w+")
            precursor_mz_file[:] = self.precursor_mz[:]
            self.precursor_mz = precursor_mz_file

        super().build_index([x["peaks"] for x in all_spectra])
        return all_spectra

    def search_identity(self, precursor_mz, peaks, ms1_tolerance_in_da, ms2_tolerance_in_da, **kwargs):
        # Determine the precursor m/z range
        precursor_mz_min = precursor_mz - ms1_tolerance_in_da
        precursor_mz_max = precursor_mz + ms1_tolerance_in_da
        spectra_idx_min = np.searchsorted(self.precursor_mz, precursor_mz_min, side='left')
        spectra_idx_max = np.searchsorted(self.precursor_mz, precursor_mz_max, side='right')
        if spectra_idx_min >= spectra_idx_max:
            return np.zeros(self.total_spectra_num, dtype=np.float32)
        else:
            entropy_similarity = self.search(peaks=peaks, ms2_tolerance_in_da=ms2_tolerance_in_da,
                                             search_type=1, search_spectra_idx_min=spectra_idx_min, search_spectra_idx_max=spectra_idx_max, **kwargs)
            return entropy_similarity

    def search_open(self, peaks, ms2_tolerance_in_da, **kwargs):
        entropy_similarity = self.search(peaks=peaks, ms2_tolerance_in_da=ms2_tolerance_in_da, search_type=0, **kwargs)
        return entropy_similarity

    def search_with_multiple_cores(self, precursor_mz_list, peaks_list, ms1_tolerance_in_da, ms2_tolerance_in_da, cores):
        self.move_index_array_to_shared_memory()

        result_mp = [np.empty((len(peaks_list), self.total_spectra_num), dtype=np.float32), 0]

        def func_merge(_, result_new):
            result_mp[0][result_mp[1], :] = result_new
            result_mp[1] += 1
            return None

        if precursor_mz_list is None:
            mp_runner = MPRunner(func_run=self.search_open, func_merge=func_merge,
                                 para_share=(ms2_tolerance_in_da,),
                                 threads=cores,
                                 max_job_in_queue=100,
                                 output_result_in_order=True)
            for peaks in peaks_list:
                mp_runner.add_job(peaks)
        else:
            mp_runner = MPRunner(func_run=self.search_identity, func_merge=func_merge,
                                 para_share=(ms1_tolerance_in_da, ms2_tolerance_in_da),
                                 threads=cores,
                                 max_job_in_queue=100,
                                 output_result_in_order=True)
            for precursor_mz, peaks in zip(precursor_mz_list, peaks_list):
                mp_runner.add_job(precursor_mz, peaks)

        mp_runner.wait_for_result()
        result, i = result_mp
        assert i == len(result)
        return result

    def move_index_array_to_shared_memory(self):
        if self.is_index_in_shared_memory:
            return

        super().move_index_array_to_shared_memory()
        if len(self.precursor_mz) == 0:
            return

        self.precursor_mz = convert_numpy_array_to_shared_memory(self.precursor_mz)

    def write_to_file(self, fo=None):
        if self.use_file_array:
            self.precursor_mz.flush()
        else:
            write_numpy_array_to_file_stream(fo, self.precursor_mz)
        super().write_to_file(fo)

    def read_from_file(self, fi=None, use_memmap=False):
        if self.use_file_array:
            self.precursor_mz = np.memmap(filename=self.path_array/"precursor_mz.npy", dtype=np.float32, mode="r")
        else:
            self.precursor_mz = read_numpy_array_from_file_stream(fi, use_memmap)
        super().read_from_file(fi, use_memmap)


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
