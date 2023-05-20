#!/usr/bin/env python3
import datetime
import os
import platform
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from ms_entropy import FlashEntropySearch, clean_spectrum
from ms_entropy.file_io.msp_file import read_one_spectrum

URL_MONA = r'https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/03d5a22c-c1e1-4101-ac70-9a4eae437ef5'
MS1_TOLERANCE_IN_DA = 0.01
MS2_TOLERANCE_IN_DA = 0.02
QUERY_SPECTRA_NUM = 100
LIBRARY_SPECTRA_NUM = 1000000


def main():
    print(f'Python version: {platform.python_version()}')
    print(f'Numpy version: {np.__version__}')
    print(f'OS: {platform.platform()}')
    print(f'CPU name: {get_processor_name()}')
    print(f'OS name: {get_os_name()}')

    path_data = Path(__file__).parent / 'data'
    # file_output = Path(sys.argv[1]) if len(sys.argv) > 1 else path_data / 'result.csv'

    print('Download library spectra and parse them.')
    # Download MassBank.us spectra.
    file_mona = download_mona(path_data)

    # Split spectra by ion mode.
    all_raw_spectra = {'P': [], 'N': []}
    for spec in read_one_spectrum(file_mona):
        # If there is no precursor_mz (which might be the GC-MS spectra), or the precursor_mz format is wrong, skip.
        try:
            spec['precursor_mz'] = float(spec.pop("precursormz"))
        except:
            continue
        ion_mode = spec.get("ion_mode", "")
        if ion_mode in all_raw_spectra:
            all_raw_spectra[ion_mode].append(spec)
            # if len(all_raw_spectra[ion_mode]) > 100:
            #     break
    print(f"Total number of spectra: {len(all_raw_spectra['P']) + len(all_raw_spectra['N'])}.")

    ###########################################################################
    result = []
    for ion_mode in all_raw_spectra:
        ########################################################################################
        print(f"Generating {LIBRARY_SPECTRA_NUM} library spectra and {QUERY_SPECTRA_NUM} query spectra for ion mode {ion_mode}.")
        # clean the spectra_library.
        spectra_library_all = []
        for spec in all_raw_spectra[ion_mode]:
            spec["peaks"] = clean_spectrum(spectrum=spec["peaks"], noise_threshold=0.01, max_mz=spec["precursor_mz"]-1.6)
            if len(spec["peaks"]) > 0:
                spectra_library_all.append(spec)

        # Generate library spectra.
        spectra_library_all = np.random.choice(spectra_library_all, LIBRARY_SPECTRA_NUM)
        spectral_library_length = len(spectra_library_all)

        # Generate query spectra.
        spectra_query_all = np.random.choice(spectra_library_all, QUERY_SPECTRA_NUM)

        ########################################################################################
        # For flash entropy search.
        entropy_search = FlashEntropySearch()
        entropy_search.build_index(spectra_library_all, clean_spectra=False)

        print(f"Start benchmarking for ion mode {ion_mode}.")
        # Step 3: Benchmark the similarity calculation.
        result = {
            "identity_search": np.zeros(QUERY_SPECTRA_NUM),
            "open_search": np.zeros(QUERY_SPECTRA_NUM),
            "neutral_loss_search": np.zeros(QUERY_SPECTRA_NUM),
            "hybrid_search": np.zeros(QUERY_SPECTRA_NUM)
        }
        for i, spec_query in enumerate(spectra_query_all):
            spec_query_peaks = spec_query["peaks"]

            # For identity search.
            start_time = time.time()
            similarity_identity_search = entropy_search.identity_search(
                precursor_mz=spec_query["precursor_mz"],
                peaks=spec_query_peaks, ms1_tolerance_in_da=MS1_TOLERANCE_IN_DA, ms2_tolerance_in_da=MS2_TOLERANCE_IN_DA, target="cpu")
            used_time = time.time() - start_time
            result["identity_search"][i] = used_time

            # For open search.
            start_time = time.time()
            similarity_entropy_search = entropy_search.open_search(peaks=spec_query_peaks, ms2_tolerance_in_da=MS2_TOLERANCE_IN_DA, target="cpu")
            used_time = time.time() - start_time
            result["open_search"][i] = used_time

            # For neutral loss search.
            start_time = time.time()
            similarity_neutral_loss_search = entropy_search.neutral_loss_search(
                precursor_mz=spec_query["precursor_mz"],
                peaks=spec_query_peaks, ms1_tolerance_in_da=MS1_TOLERANCE_IN_DA, ms2_tolerance_in_da=MS2_TOLERANCE_IN_DA, target="cpu")
            used_time = time.time() - start_time
            result["neutral_loss_search"][i] = used_time

            # For hybrid search.
            start_time = time.time()
            similarity_hybrid_search = entropy_search.hybrid_search(
                precursor_mz=spec_query["precursor_mz"],
                peaks=spec_query_peaks, ms1_tolerance_in_da=MS1_TOLERANCE_IN_DA, ms2_tolerance_in_da=MS2_TOLERANCE_IN_DA, target="cpu")
            used_time = time.time() - start_time
            result["hybrid_search"][i] = used_time

        # Step 4: print the result.
        print(f"....The median time for identity search is {np.median(result['identity_search'])*1000} milliseconds.")
        print(f"....The median time for open search is {np.median(result['open_search'])*1000} milliseconds.")
        print(f"....The median time for neutral loss search is {np.median(result['neutral_loss_search'])*1000} milliseconds.")
        print(f"....The median time for hybrid search is {np.median(result['hybrid_search'])*1000} milliseconds.")

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command.split()).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""

def get_os_name():
    if platform.system() == "Linux":
        command = "cat /etc/os-release"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "PRETTY_NAME" in line:
                return re.sub( ".*PRETTY_NAME.*=", "", line,1)
    elif platform.system() == "Darwin":
        command = "system_profiler SPSoftwareDataType | grep 'System Version'"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        return re.sub( ".*System Version: ", "", all_info,1)
    elif platform.system() == "Windows":
        return platform.platform()


def download_mona(path_data):
    # Download MassBank.us spectra.
    path_data.mkdir(parents=True, exist_ok=True)
    file_mona = path_data / f'mona-{datetime.date.today()}.zip'

    if not file_mona.exists():
        print(f'Downloading {URL_MONA} to {file_mona}')
        with urllib.request.urlopen(URL_MONA) as response:
            data = response.read()
            with open(file_mona, 'wb') as f:
                f.write(data)
    return file_mona


if __name__ == '__main__':
    main()
