import pickle
import numpy as np
import struct
import pandas as pd

from .entropy_search import EntropySearch, write_numpy_array_to_file_stream, read_numpy_array_from_file_stream
from mimas.spectra.similarity import clean_spectrum


class EntropySearchLibrary:
    def __init__(self) -> None:
        self.parameter = {
            "remove_precursor_peak": 1.6,
            "noise_threshold": 0.01,
            "ms2_tolerance_in_da_for_clean_spectrum": 0.05,
        }
        self.raw_spectral_library = []
        self.metadata: pd.DataFrame = None
        self.entropy_search_library: dict[int, dict] = {}

    def search_identity(self, *, charge, precursor_mz, peaks, ms1_tolerance_in_da, ms2_tolerance_in_da, **kwargs):
        if charge in self.entropy_search_library:
            result = self.entropy_search_library[charge]["product_ion"].search_identity(
                precursor_mz=precursor_mz, peaks=peaks, ms1_tolerance_in_da=ms1_tolerance_in_da, ms2_tolerance_in_da=ms2_tolerance_in_da)
        else:
            result = np.zeros(0, dtype=np.float32)
        return result

    def search_open(self, *, charge, peaks, ms2_tolerance_in_da, **kwargs):
        if charge in self.entropy_search_library:
            result = self.entropy_search_library[charge]["product_ion"].search_open(
                peaks=peaks, ms2_tolerance_in_da=ms2_tolerance_in_da)
        else:
            result = np.zeros(0, dtype=np.float32)
        return result

    def search_neutral_loss(self, *, charge, precursor_mz, peaks, ms2_tolerance_in_da, **kwargs):
        if charge in self.entropy_search_library:
            peaks_neutral_loss = np.copy(peaks)
            peaks_neutral_loss[:, 0] = precursor_mz-peaks_neutral_loss[:, 0]
            peaks_neutral_loss = np.ascontiguousarray(np.flip(peaks_neutral_loss, axis=0))
            result = self.entropy_search_library[charge]["neutral_loss"].search_open(
                peaks=peaks_neutral_loss, ms2_tolerance_in_da=ms2_tolerance_in_da)
        else:
            result = np.zeros(0, dtype=np.float32)
        return result

    def add_spectrum_to_library(self, spectrum_info: dict):
        """
        Add a spectrum for building the library.

        Meaning of the return value:
        0: spectrum is added to the library
        1: The precursor m/z is too high (higher than 2000), and the spectrum is not added to the library.
        2: The precursor m/z is less than 0, and the spectrum is not added to the library.
        3: The peaks are empty, and the spectrum is not added to the library.
        4: The peaks after cleaning are empty, and the spectrum is not added to the library.

        :param spectrum_info: A dictionary of spectrum information.
            The dictionary should have the following keys:
            "precursor_mz": The precursor m/z.
            "peaks": The peaks of the spectrum.
            "charge": The precursor charge.
            The rest information will be storaged as the metadata of the spectrum.
        :return: int
        """
        if spectrum_info["precursor_mz"] >= 2000.:
            return 1

        if spectrum_info["precursor_mz"] < 0.:
            return 2

        if len(spectrum_info["peaks"]) <= 0:
            return 3

        # Clean the peaks
        precursor_mz, peaks = spectrum_info["precursor_mz"], spectrum_info["peaks"]
        remove_precursor_peak = self.parameter["remove_precursor_peak"]
        if remove_precursor_peak > 0:
            max_mz = precursor_mz-remove_precursor_peak
        else:
            max_mz = None

        peaks_clean = clean_spectrum(peaks,
                                     max_mz=max_mz,
                                     noise_threshold=self.parameter["noise_threshold"],
                                     remove_isotope=True,
                                     normalize_intensity=True,
                                     ms2_da=self.parameter["ms2_tolerance_in_da_for_clean_spectrum"])
        spectrum_info["peaks_clean"] = peaks_clean
        spectrum_info.pop("peaks", None)

        if len(spectrum_info["peaks_clean"]) <= 0:
            return 4

        charge = spectrum_info["charge"]
        if charge not in self.entropy_search_library:
            self.entropy_search_library[charge] = {}
        self.raw_spectral_library.append(spectrum_info)
        return 0

    def build_index(self):
        for charge in self.entropy_search_library:
            all_spectra_product_ion = []
            for idx, spectrum in enumerate(self.raw_spectral_library):
                if spectrum["charge"] == charge:
                    all_spectra_product_ion.append({
                        "idx": idx,
                        "precursor_mz": spectrum["precursor_mz"],
                        "peaks": spectrum["peaks_clean"],
                    })

            # Generate the index for product ions
            entropy_search_product_ion = EntropySearch()
            all_spectra_product_ion = entropy_search_product_ion.build_index(all_spectra_product_ion, sort_by_precursor_mz=True)
            result_idx = np.array([spectrum["idx"] for spectrum in all_spectra_product_ion], dtype=np.uint64)

            all_spectra_neutral_loss = []
            for spectrum in all_spectra_product_ion:
                peaks_neutral_loss = np.copy(spectrum["peaks"])
                peaks_neutral_loss[:, 0] = spectrum["precursor_mz"]-peaks_neutral_loss[:, 0]
                peaks_neutral_loss = np.flip(peaks_neutral_loss, axis=0)
                all_spectra_neutral_loss.append({
                    "precursor_mz": spectrum["precursor_mz"],
                    "peaks": peaks_neutral_loss,
                })

            # Generate the index for neutral losses
            entropy_search_neutral_loss = EntropySearch()
            all_spectra_neutral_loss = entropy_search_neutral_loss.build_index(all_spectra_neutral_loss, sort_by_precursor_mz=False)

            # Record the original peaks
            origin_peaks = [x["peaks"] for x in all_spectra_product_ion]
            peak_length = np.array([len(x) for x in origin_peaks], dtype=np.int32)
            origin_peaks = np.concatenate(origin_peaks, dtype=np.float32)
            peak_length = np.cumsum(peak_length)
            self.entropy_search_library[charge] = {
                "result_idx": result_idx,
                "origin_peaks": (peak_length, origin_peaks),
                "product_ion": entropy_search_product_ion,
                "neutral_loss": entropy_search_neutral_loss,
            }

        # Index the metadata
        [x.pop("peaks_clean", None) for x in self.raw_spectral_library]
        df = pd.DataFrame(self.raw_spectral_library)
        self.metadata = df

    def write_to_file(self, filename):
        fo = open(filename, "wb")
        fo.write(struct.pack("QQ", 0, 0))
        file_info = {
            "parameter": self.parameter,
            "charges": list(self.entropy_search_library.keys()),
            "metadata": [fo.tell(), 0],
        }
        # Write the raw spectral library information to file
        metadata_bytes = pickle.dumps(self.metadata)
        fo.write(metadata_bytes)
        file_info["metadata"][1] = fo.tell()-file_info["metadata"][0]

        # Write the library to file
        for charge in file_info["charges"]:
            data = self.entropy_search_library[charge]
            write_numpy_array_to_file_stream(fo, data["result_idx"])
            write_numpy_array_to_file_stream(fo, data["origin_peaks"][0])
            write_numpy_array_to_file_stream(fo, data["origin_peaks"][1])
            data["product_ion"].write_to_file(fo)
            data["neutral_loss"].write_to_file(fo)

        # Write all the information of the library to file
        location = fo.tell()
        file_info_data = pickle.dumps(file_info)
        fo.write(file_info_data)
        fo.seek(0)
        fo.write(struct.pack("QQ", location, len(file_info_data)))

    def read_from_file(self, filename, use_memmap):
        fi = open(filename, "rb")
        file_info_location, file_info_length = struct.unpack("QQ", fi.read(struct.calcsize('QQ')))
        fi.seek(file_info_location)
        file_info = pickle.loads(fi.read(file_info_length))

        # Read the raw spectral library location information
        fi.seek(file_info["metadata"][0])
        data = fi.read(file_info["metadata"][1])
        self.metadata = pickle.loads(data)

        # Read the library from file
        for charge in file_info["charges"]:
            data = {
                "product_ion": EntropySearch(),
                "neutral_loss": EntropySearch(),
            }
            data["result_idx"] = read_numpy_array_from_file_stream(fi, use_memmap)
            data["origin_peaks"] = (read_numpy_array_from_file_stream(fi, use_memmap), read_numpy_array_from_file_stream(fi, use_memmap))
            data["product_ion"].read_from_file(fi, use_memmap)
            data["neutral_loss"].read_from_file(fi, use_memmap)
            self.entropy_search_library[charge] = data

    def get_metadata(self, charge, idx):
        library_idx = self.entropy_search_library[charge]["result_idx"][idx]
        return self.metadata.iloc[library_idx].to_dict()
