import shutil
import numpy as np
from pathlib import Path
from library.entropy_search import EntropySearch, apply_weight_to_intensity
from library.hybrid_search import EntropyHybridSearchCore


class EntropySearchUniverse:
    def __init__(self, path_data, mode) -> None:
        self.path_data = Path(str(path_data))
        self.entropy_search_library: dict[int, dict] = {}
        if mode == "write":
            # shutil.rmtree(self.path_data, ignore_errors=True)
            self.path_data.mkdir(parents=True, exist_ok=True)
        elif mode == "read":
            for charge in self.path_data.glob("charge_*"):
                self.entropy_search_library[int(charge.name.split("_")[-1])] = None

    def load_entropy_search(self, charge):
        if charge in self.entropy_search_library:
            if self.entropy_search_library[charge] is None:
                path_index = self.path_data / f"charge_{charge}"
                result_idx = np.memmap(path_index / "result_idx.npy", dtype=np.uint64, mode="r")

                entropy_search_product_ion = EntropySearch(path_array=path_index / "product_ion")
                entropy_search_product_ion.read_from_file(use_memmap=True)

                entropy_search_neutral_loss = EntropySearch(path_array=path_index / "neutral_loss")
                entropy_search_neutral_loss.read_from_file(use_memmap=True)

                self.entropy_search_library[charge] = {"product_ion": entropy_search_product_ion,
                                                       "neutral_loss": entropy_search_neutral_loss,
                                                       "result_idx": result_idx}
            return True
        else:
            return False

    def search_identity(self, *, charge, precursor_mz, peaks, ms1_tolerance_in_da, ms2_tolerance_in_da, **kwargs):
        if self.load_entropy_search(charge):
            result_similarity = self.entropy_search_library[charge]["product_ion"].search_identity(
                precursor_mz=precursor_mz, peaks=peaks, ms1_tolerance_in_da=ms1_tolerance_in_da, ms2_tolerance_in_da=ms2_tolerance_in_da)
            return result_similarity
        else:
            return np.zeros(0, dtype=np.float32)

    def search_open(self, *, charge, peaks, ms2_tolerance_in_da, **kwargs):
        if self.load_entropy_search(charge):
            result_similarity = self.entropy_search_library[charge]["product_ion"].search_open(
                peaks=peaks, ms2_tolerance_in_da=ms2_tolerance_in_da, **kwargs)
            return result_similarity
        else:
            return np.zeros(0, dtype=np.float32)

    def search_neutral_loss(self, *, charge, precursor_mz, peaks, ms2_tolerance_in_da, **kwargs):
        if self.load_entropy_search(charge):
            peaks_neutral_loss = np.copy(peaks)
            peaks_neutral_loss[:, 0] = precursor_mz-peaks_neutral_loss[:, 0]
            peaks_neutral_loss = np.ascontiguousarray(np.flip(peaks_neutral_loss, axis=0))

            result_similarity = self.entropy_search_library[charge]["neutral_loss"].search_open(
                peaks=peaks_neutral_loss, ms2_tolerance_in_da=ms2_tolerance_in_da, **kwargs)
            return result_similarity
        else:
            return np.zeros(0, dtype=np.float32)

    def get_metadata(self, charge, idx):
        return self.entropy_search_library[charge]["result_idx"][idx]

    def build_index_for_all_spectra_with_same_charge(self, charge, all_spectra, max_ms2_tolerance_in_da=0.024):
        """
        :param all_spectra: A dictionary of spectrum information. The spectrum information will be modified in this function.
            The dictionary should have the following keys:
            "scan": The scan number, which is used as the index of the spectrum.
            "precursor_mz": The precursor m/z.
            "peaks": The peaks of the spectrum.
        """
        path_index = self.path_data / f"charge_{charge}"
        path_index.mkdir(parents=True, exist_ok=True)

        # Generate the index for product ions
        entropy_search_product_ion = EntropySearch(path_array=path_index / "product_ion", max_ms2_tolerance_in_da=max_ms2_tolerance_in_da)
        all_spectra_product_ion = entropy_search_product_ion.build_index(all_spectra, sort_by_precursor_mz=True)
        entropy_search_product_ion.write_to_file()

        result_idx = np.memmap(path_index / "result_idx.npy", dtype=np.uint64, mode="w+", shape=(len(all_spectra_product_ion),))
        result_idx[:] = [spectrum["scan"] for spectrum in all_spectra_product_ion]
        # Write to disk
        result_idx.flush()

        for spectrum in all_spectra_product_ion:
            peaks_neutral_loss = np.copy(spectrum["peaks"])
            peaks_neutral_loss[:, 0] = spectrum["precursor_mz"]-peaks_neutral_loss[:, 0]
            peaks_neutral_loss = np.flip(peaks_neutral_loss, axis=0)
            spectrum["peaks"] = peaks_neutral_loss

        # Generate the index for neutral losses
        entropy_search_neutral_loss = EntropySearch(path_array=path_index / "neutral_loss", max_ms2_tolerance_in_da=max_ms2_tolerance_in_da)
        entropy_search_neutral_loss.build_index(all_spectra_product_ion, sort_by_precursor_mz=False)
        entropy_search_neutral_loss.write_to_file()

        self.entropy_search_library[charge] = None
        return 0

    def build_index_for_universe(self, charge, file_input, file_offset_start, file_offset_end, max_ms2_tolerance_in_da=0.024):
        from atlas.ms_spectrum import read_spectrum_from_file_stream
        path_index = self.path_data / f"charge_{charge}"
        path_index.mkdir(parents=True, exist_ok=True)

        # Get the number of spectra, the number of peaks
        total_spectra_num = 0
        total_peak_num = 0
        all_scan = []
        file_input.seek(file_offset_start)
        for spec in read_spectrum_from_file_stream(file_input, file_offset_end):
            total_spectra_num += 1
            total_peak_num += len(spec["peaks"])
            all_scan.append(spec["scan"])
        all_scan = np.array(all_scan, dtype=np.uint64)
        all_scan.tofile(path_index / "result_idx.npy")

        # total_spectra_num can not be bigger than 2^32-1 (uint32), total_peak_num can not be bigger than 2^63-1 (int64)
        assert total_spectra_num < 4294967295 and total_peak_num < 9223372036854775807

        # Generate the index for product ions
        entropy_search_product_ion = EntropySearch(path_array=path_index / "product_ion", max_ms2_tolerance_in_da=max_ms2_tolerance_in_da)
        self._build_index(entropy_search_product_ion, total_spectra_num, total_peak_num, file_input, file_offset_start, file_offset_end, is_neutral_loss=False)

        # Generate the index for neutral losses
        entropy_search_neutral_loss = EntropySearch(path_array=path_index / "neutral_loss", max_ms2_tolerance_in_da=max_ms2_tolerance_in_da)
        self._build_index(entropy_search_neutral_loss, total_spectra_num, total_peak_num, file_input, file_offset_start, file_offset_end, is_neutral_loss=True)
        return 0

    def _build_index(self, entropy_search, total_spectra_num, total_peak_num,
                     file_input, file_offset_start, file_offset_end,
                     is_neutral_loss=False, max_indexed_mz=1500.00005):

        entropy_search.precursor_mz = \
            np.memmap(filename=entropy_search.path_array/"precursor_mz.npy", shape=total_spectra_num, dtype=np.float32, mode="w+")

        peak_idx = 0
        file_input.seek(file_offset_start)
        dtype_peak_data = np.dtype([("mz", np.float32), ("intensity", np.float32), ("idx", np.uint32)], align=True)
        peak_data = np.zeros(total_peak_num, dtype=dtype_peak_data)
        for idx, spec in enumerate(read_spectrum_from_file_stream(file_input, file_offset_end)):
            entropy_search.precursor_mz[idx] = spec["precursor_mz"]
            peaks = spec["peaks"]
            assert peaks.shape[0] >= 1  # Check if the number of peaks is greater than 0.
            assert abs(np.sum(peaks[:, 1])-1) < 1e-4  # Check if the sum of all intensities is 1.
            # assert np.all(peaks[:-1, 0] <= peaks[1:, 0])  # Check if the peaks are sorted.
            # Check if the peaks are separated enough.
            assert peaks.shape[0] <= 1 or np.min(peaks[1:, 0] - peaks[:-1, 0]) > entropy_search.max_ms2_tolerance_in_da * 2, \
                f'Error found in spectrum {idx}, the peaks are too close to each other. The peaks are {str(peaks)}'

            if is_neutral_loss:
                peaks_neutral_loss = np.copy(peaks)
                peaks_neutral_loss[:, 0] = spec["precursor_mz"]-peaks_neutral_loss[:, 0]
                peaks_neutral_loss = np.flip(peaks_neutral_loss, axis=0)
                peaks = peaks_neutral_loss

            # Pre-calculate library peaks entropy
            peaks_clean = np.asarray(apply_weight_to_intensity(peaks))
            peaks_clean[:, 1] /= 2

            # Store the peaks
            peak_data.view(np.float32).reshape(-1, 3)[peak_idx:(peak_idx + peaks.shape[0]), 0] = peaks_clean[:, 0]
            peak_data.view(np.float32).reshape(-1, 3)[peak_idx:(peak_idx + peaks.shape[0]), 1] = peaks_clean[:, 1]
            peak_data.view(np.uint32).reshape(-1, 3)[peak_idx:(peak_idx + peaks.shape[0]), 2] = idx
            peak_idx += peaks.shape[0]

        # Generate the index
        peak_data.sort(order="mz")
        peak_data.view(np.float32).reshape(-1, 3)[:, 0].tofile(entropy_search.path_array / "all_peaks_mz.npy")
        peak_data.view(np.float32).reshape(-1, 3)[:, 1].tofile(entropy_search.path_array / "all_peaks_intensity.npy")
        peak_data.view(np.uint32).reshape(-1, 3)[:, 2].tofile(entropy_search.path_array / "all_peaks_spec_idx.npy")
        del peak_data

        all_peaks_mz = np.memmap(entropy_search.path_array / 'all_peaks_mz.npy', dtype=np.float32, mode='r+', shape=(total_peak_num,))
        all_peaks_intensity = np.memmap(entropy_search.path_array / 'all_peaks_intensity.npy', dtype=np.float32, mode='r+', shape=(total_peak_num,))
        all_peaks_spec_idx = np.memmap(entropy_search.path_array / 'all_peaks_spec_idx.npy', dtype=np.uint32, mode='r+', shape=(total_peak_num,))

        max_mz = min(np.max(all_peaks_mz), max_indexed_mz)
        search_array = np.arange(0., max_mz, entropy_search.mz_index_step)
        all_peaks_mz_idx_start = np.searchsorted(all_peaks_mz, search_array, side='left').astype(np.int64)
        # Write all_peaks_mz_idx_start to file for memory mapping
        all_peaks_mz_idx_start.tofile(entropy_search.path_array / 'all_peaks_mz_idx_start.npy')
        all_peaks_mz_idx_start = np.memmap(entropy_search.path_array / 'all_peaks_mz_idx_start.npy', dtype=np.int64, mode='r')

        entropy_search.index = [all_peaks_mz_idx_start, all_peaks_mz, all_peaks_intensity, all_peaks_spec_idx]
        entropy_search.total_spectra_num = total_spectra_num
        entropy_search.write_to_file()
        return entropy_search


class EntropySearchUniverseIdentity:
    def __init__(self, path_data, mode) -> None:
        self._mz_index_step = 0.01
        self.path_data = Path(str(path_data))
        self.entropy_search_library: dict[int, dict] = {}
        if mode == "write":
            shutil.rmtree(self.path_data, ignore_errors=True)
            self.path_data.mkdir(parents=True, exist_ok=True)
        elif mode == "read":
            for charge in self.path_data.glob("charge_*"):
                self.entropy_search_library[int(charge.name.split("_")[-1])] = {}
                for precursor_mz in charge.glob("precursor_mz_bin_*"):
                    self.entropy_search_library[int(charge.name.split("_")[-1])][int(precursor_mz.name.split("_")[-1])] = None

    def load_entropy_search(self, charge, precursor_mz_bin):
        if charge in self.entropy_search_library and precursor_mz_bin in self.entropy_search_library[charge]:
            if self.entropy_search_library[charge][precursor_mz_bin] is None:
                path_index = self.path_data / f"charge_{charge}" / f"precursor_mz_bin_{precursor_mz_bin}"
                result_idx = np.memmap(path_index / "result_idx.npy", dtype=np.uint64, mode="r")

                entropy_search_product_ion = EntropySearch(path_array=path_index, mz_index_step=self._mz_index_step)
                entropy_search_product_ion.read_from_file()

                self.entropy_search_library[charge][precursor_mz_bin] = {
                    "product_ion": entropy_search_product_ion,
                    "result_idx": result_idx
                }
            return True
        else:
            return False

    def search_identity(self, *, charge, precursor_mz_bin, precursor_mz, peaks, ms1_tolerance_in_da, ms2_tolerance_in_da, **kwargs):
        if self.load_entropy_search(charge, precursor_mz_bin):
            result_similarity = self.entropy_search_library[charge][precursor_mz_bin]["product_ion"].search_identity(
                precursor_mz=precursor_mz, peaks=peaks, ms1_tolerance_in_da=ms1_tolerance_in_da, ms2_tolerance_in_da=ms2_tolerance_in_da, **kwargs)
            self.entropy_search_library[charge][precursor_mz_bin] = None
            return result_similarity
        else:
            return np.zeros(0, dtype=np.int64)

    def build_index(self, charge, precursor_mz_bin, all_spectra):
        """
        :param all_spectra: A dictionary of spectrum information. The spectrum information will be modified in this function.
            The dictionary should have the following keys:
            "scan": The scan number, which is used as the index of the spectrum.
            "precursor_mz": The precursor m/z.
            "peaks": The peaks of the spectrum.
        """
        path_index = self.path_data / f"charge_{charge}" / f"precursor_mz_bin_{precursor_mz_bin}"
        path_index.mkdir(parents=True, exist_ok=True)

        # Generate the index for product ions
        entropy_search_product_ion = EntropySearch(path_array=path_index, mz_index_step=self._mz_index_step)
        all_spectra_product_ion = entropy_search_product_ion.build_index(all_spectra, sort_by_precursor_mz=True)
        entropy_search_product_ion.write_to_file()
        result_idx = np.memmap(path_index / "result_idx.npy", dtype=np.uint64, mode="w+", shape=(len(all_spectra_product_ion),))
        result_idx[:] = [spectrum["scan"] for spectrum in all_spectra_product_ion]
        result_idx.flush()
        return 0

    def get_metadata(self, charge, precursor_mz, idx):
        return self.entropy_search_library[charge][int(precursor_mz)]["result_idx"][idx]


class EntropySearchUniverseHybrid:
    def __init__(self, path_data, mode) -> None:
        self.path_data = Path(str(path_data))
        self.entropy_search_library: dict[int, dict] = {}
        if mode == "write":
            # shutil.rmtree(self.path_data, ignore_errors=True)
            self.path_data.mkdir(parents=True, exist_ok=True)
        elif mode == "read":
            for charge in self.path_data.glob("charge_*"):
                self.entropy_search_library[int(charge.name.split("_")[-1])] = None

    def build_index_for_universe(self, charge, file_input, file_offset_start, file_offset_end, max_ms2_tolerance_in_da=0.024):
        path_index = self.path_data / f"charge_{charge}"
        path_index.mkdir(parents=True, exist_ok=True)

        # Get the number of spectra, the number of peaks
        total_spectra_num = 0
        total_peak_num = 0
        all_scan = []
        file_input.seek(file_offset_start)
        for spec in read_spectrum_from_file_stream(file_input, file_offset_end):
            total_spectra_num += 1
            total_peak_num += len(spec["peaks"])
            all_scan.append(spec["scan"])
        all_scan = np.array(all_scan, dtype=np.uint64)
        all_scan.tofile(path_index / "result_idx.npy")

        # total_spectra_num can not be bigger than 2^32-1 (uint32), total_peak_num can not be bigger than 2^63-1 (int64)
        assert total_spectra_num < 4294967295 and total_peak_num < 9223372036854775807

        # Generate the index for product ions
        entropy_hybird_search = EntropyHybridSearchCore(path_array=path_index / "hybrid", max_ms2_tolerance_in_da=max_ms2_tolerance_in_da)
        self._build_index(entropy_hybird_search, total_spectra_num, total_peak_num, file_input, file_offset_start, file_offset_end)
        return 0

    def _build_index(self, entropy_search, total_spectra_num, total_peak_num,
                     file_input, file_offset_start, file_offset_end,
                     max_indexed_mz=1500.00005):
        entropy_search.precursor_mz = \
            np.memmap(filename=entropy_search.path_array/"precursor_mz.npy", shape=total_spectra_num, dtype=np.float32, mode="w+")

        peak_idx = 0
        file_input.seek(file_offset_start)
        dtype_len = 6
        dtype_peak_data = np.dtype([("mz", np.float32),
                                    ("nl_mz", np.float32),
                                    ("intensity", np.float32),
                                    ("spec_idx", np.uint32),
                                    ("peak_idx", np.uint64)], align=True)
        peak_data = np.zeros(total_peak_num, dtype=dtype_peak_data)
        for idx, spec in enumerate(read_spectrum_from_file_stream(file_input, file_offset_end)):
            entropy_search.precursor_mz[idx] = spec["precursor_mz"]
            peaks = spec["peaks"]
            assert peaks.shape[0] >= 1  # Check if the number of peaks is greater than 0.
            assert abs(np.sum(peaks[:, 1])-1) < 1e-4  # Check if the sum of all intensities is 1.
            # assert np.all(peaks[:-1, 0] <= peaks[1:, 0])  # Check if the peaks are sorted.
            # Check if the peaks are separated enough.
            assert peaks.shape[0] <= 1 or np.min(peaks[1:, 0] - peaks[:-1, 0]) > entropy_search.max_ms2_tolerance_in_da * 2, \
                f'Error found in spectrum {idx}, the peaks are too close to each other. The peaks are {str(peaks)}'

            # Pre-calculate library peaks entropy
            peaks_clean = np.asarray(apply_weight_to_intensity(peaks))
            peaks_clean[:, 1] /= 2

            # Store the peaks
            neutral_loss_mz = spec["precursor_mz"]-peaks_clean[:, 0]
            # Store the peaks
            peak_data.view(np.float32).reshape(-1, dtype_len)[peak_idx:(peak_idx + peaks.shape[0]), 0] = peaks_clean[:, 0]
            peak_data.view(np.float32).reshape(-1, dtype_len)[peak_idx:(peak_idx + peaks.shape[0]), 1] = neutral_loss_mz
            peak_data.view(np.float32).reshape(-1, dtype_len)[peak_idx:(peak_idx + peaks.shape[0]), 2] = peaks_clean[:, 1]
            peak_data.view(np.uint32).reshape(-1, dtype_len)[peak_idx:(peak_idx + peaks.shape[0]), 3] = idx
            peak_idx += peaks.shape[0]
        assert peak_idx == total_peak_num

        # Generate the index for product ions
        peak_data.sort(order="mz")
        peak_data.view(np.uint64).reshape(-1, dtype_len//2)[:, 2] = np.arange(0, total_peak_num, dtype=np.uint64)
        peak_data.view(np.float32).reshape(-1, dtype_len)[:, 0].tofile(entropy_search.path_array / "all_peaks_mz.npy")
        peak_data.view(np.float32).reshape(-1, dtype_len)[:, 2].tofile(entropy_search.path_array / "all_peaks_intensity.npy")
        peak_data.view(np.uint32).reshape(-1, dtype_len)[:, 3].tofile(entropy_search.path_array / "all_peaks_spec_idx.npy")

        all_peaks_mz = np.memmap(entropy_search.path_array / 'all_peaks_mz.npy', dtype=np.float32, mode='r', shape=(total_peak_num,))
        all_peaks_intensity = np.memmap(entropy_search.path_array / 'all_peaks_intensity.npy', dtype=np.float32, mode='r', shape=(total_peak_num,))
        all_peaks_spec_idx = np.memmap(entropy_search.path_array / 'all_peaks_spec_idx.npy', dtype=np.uint32, mode='r', shape=(total_peak_num,))

        # Build the index for all_peaks_mz
        max_mz = min(np.max(all_peaks_mz), max_indexed_mz)
        search_array = np.arange(0., max_mz, entropy_search.mz_index_step)
        all_peaks_mz_idx_start = np.searchsorted(all_peaks_mz, search_array, side='left').astype(np.int64)
        all_peaks_mz_idx_start.tofile(entropy_search.path_array / 'all_peaks_mz_idx_start.npy')
        all_peaks_mz_idx_start = np.memmap(entropy_search.path_array / 'all_peaks_mz_idx_start.npy', dtype=np.int64, mode='r', shape=(len(search_array),))

        # Build the index for neutral loss
        peak_data.sort(order="nl_mz")
        peak_data.view(np.float32).reshape(-1, dtype_len)[:, 1].tofile(entropy_search.path_array / "all_nl_mz.npy")
        peak_data.view(np.float32).reshape(-1, dtype_len)[:, 2].tofile(entropy_search.path_array / "all_nl_intensity.npy")
        peak_data.view(np.uint32).reshape(-1, dtype_len)[:, 3].tofile(entropy_search.path_array / "all_nl_spec_idx.npy")
        peak_data.view(np.uint64).reshape(-1, dtype_len//2)[:, 2].tofile(entropy_search.path_array / "all_peaks_idx_for_nl.npy")
        del peak_data

        all_nl_mz = np.memmap(entropy_search.path_array / 'all_nl_mz.npy', dtype=np.float32, mode='r', shape=(total_peak_num,))
        all_nl_intensity = np.memmap(entropy_search.path_array / 'all_nl_intensity.npy', dtype=np.float32, mode='r', shape=(total_peak_num,))
        all_nl_spec_idx = np.memmap(entropy_search.path_array / 'all_nl_spec_idx.npy', dtype=np.uint32, mode='r', shape=(total_peak_num,))
        all_peaks_idx_for_nl = np.memmap(entropy_search.path_array / 'all_peaks_idx_for_nl.npy', dtype=np.uint64, mode='r', shape=(total_peak_num,))

        # Build the index for all_nl_mz
        max_mz = min(np.max(all_nl_mz), max_indexed_mz)
        search_array = np.arange(0., max_mz, entropy_search.mz_index_step)
        all_nl_mz_idx_start = np.searchsorted(all_nl_mz, search_array, side='left').astype(np.int64)
        all_nl_mz_idx_start.tofile(entropy_search.path_array / 'all_nl_mz_idx_start.npy')
        all_nl_mz_idx_start = np.memmap(entropy_search.path_array / 'all_nl_mz_idx_start.npy', dtype=np.int64, mode='r', shape=(len(search_array),))

        entropy_search.index = [all_peaks_mz_idx_start, all_peaks_mz, all_peaks_intensity, all_peaks_spec_idx,
                                all_nl_mz_idx_start, all_nl_mz, all_nl_intensity, all_nl_spec_idx, all_peaks_idx_for_nl]
        entropy_search.total_spectra_num = total_spectra_num
        entropy_search.write_to_file()
        return entropy_search

    def load_entropy_search(self, charge):
        if charge in self.entropy_search_library:
            if self.entropy_search_library[charge] is None:
                path_index = self.path_data / f"charge_{charge}"
                result_idx = np.memmap(path_index / "result_idx.npy", dtype=np.uint64, mode="r")

                entropy_search = EntropyHybridSearchCore(path_array=path_index / "hybrid")
                entropy_search.read_from_file(use_memmap=True)

                self.entropy_search_library[charge] = {"hybrid": entropy_search,
                                                       "result_idx": result_idx}
            return True
        else:
            return False

    def search_hybrid(self, *, charge, precursor_mz, peaks, ms2_tolerance_in_da, target, **kwargs):
        if self.load_entropy_search(charge):
            result_similarity = self.entropy_search_library[charge]["hybrid"].search(
                precursor_mz=precursor_mz, peaks=peaks, ms2_tolerance_in_da=ms2_tolerance_in_da, target=target)
            return result_similarity
        else:
            return np.zeros(0, dtype=np.float32)
