#!/usr/bin/env python3
import pickle
import time
import datetime
import urllib.request
from pathlib import Path
import numpy as np

from flash_entropy import mgf_file, msp_file
from flash_entropy.entropy_search import EntropySearch
from flash_entropy.functions import clean_spectrum


url_mona = r'https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/03d5a22c-c1e1-4101-ac70-9a4eae437ef5'


def main():
    path_data = Path(__file__).parent / 'data'
    file_index = path_data / 'index-neutral_loss.pkl'
    all_entropy_search = load_library_index(path_data, file_index)

    ###########################################################################
    # Randomly select a spectrum from the negative library as the query.
    ion_mode = 'N'
    spec_query = {
        'precursor_mz': 611.2246,
        'peaks': np.array([[205.0351, 78.571429], [206.0434, 61.764706], [380.1302, 100.000000], [423.1333, 34.033613]], dtype=np.float32),
    }
    # Clean the spectrum first.
    spec_query['peaks'] = clean_spectrum(peaks=spec_query['peaks'], max_mz=spec_query['precursor_mz']-1.6, noise_threshold=0.01, ms2_da=0.05)
    # Transform the spectrum to neutral loss.
    spec_query['peaks'] = neutral_loss_transform(precursor_mz=spec_query['precursor_mz'], peaks=spec_query['peaks'])
    
    ###########################################################################
    # Neutral Loss Search with Flash Entropy Search.
    print('Neutral Loss Search with Flash Entropy Search')
    start = time.time()
    open_result = all_entropy_search[ion_mode].search_open(peaks=spec_query['peaks'], ms2_tolerance_in_da=0.02)
    print(f'Finished neutral loss search with Flash Entropy Search in {time.time() - start:.4f} seconds with {len(open_result)} results.')
    # Find the top 5 matches.
    print('Top 5 matches:')
    matches_id = np.argsort(open_result)[::-1]
    for i in range(5):
        print(f'Rank {i+1}: {all_entropy_search[ion_mode].library_id[matches_id[i]]} with score {open_result[matches_id[i]]:.4f}')


def neutral_loss_transform(precursor_mz, peaks):
    peaks[:, 0] = precursor_mz - peaks[:, 0]
    # Sort the peaks by m/z.
    peaks = peaks[np.argsort(peaks[:, 0])]
    return peaks


def load_library_index(path_data, file_index):
    # If the index does not exist, build it.
    if not file_index.exists():
        # Download MassBank.us spectra.
        file_mona = download_mona(path_data)

        # Load spectra from NoNA data
        print(f'Loading spectra from {file_mona}, this may take a while.')
        all_spectra = reading_spectra_from_mona(str(file_mona))

        # Build the index.
        print('Building index, this will only need to be done once.')
        all_entropy_search = {}
        for ion_mode in all_spectra:
            print(f'Building index for spectra with ion mode {ion_mode}')
            spectra = all_spectra[ion_mode]
            entropy_search = EntropySearch()  # Create a new EntropySearch object.
            spectra = entropy_search.build_index(spectra)  # Build the index.
            entropy_search.library_id = [spec['id'] for spec in spectra]  # Record the library IDs.
            all_entropy_search[ion_mode] = entropy_search

        # Save the index.
        print('Saving index')
        with open(file_index, 'wb') as f:
            pickle.dump(all_entropy_search, f)
    else:
        # Load the index.
        print('Loading index')
        with open(file_index, 'rb') as f:
            all_entropy_search = pickle.load(f)
    return all_entropy_search


def download_mona(path_data):
    # Download MassBank.us spectra.
    path_data.mkdir(parents=True, exist_ok=True)
    file_mona = path_data / f'mona-{datetime.date.today()}.zip'

    if not file_mona.exists():
        print(f'Downloading {url_mona} to {file_mona}')
        with urllib.request.urlopen(url_mona) as response:
            data = response.read()
            with open(file_mona, 'wb') as f:
                f.write(data)
    return file_mona


def reading_spectra_from_mona(file_mona):
    all_spectra = {}
    for i, spec in enumerate(msp_file.read_one_spectrum(file_mona)):
        if i % 1000 == 0:
            print(f'Total read {i} spectra', end='\r')
        ion_mode = spec.get("ion_mode", "")
        if ion_mode in {'P', 'N'}:
            try:
                # Read peaks and precursor m/z, then clean the spectrum.
                precursor_mz = float(spec['precursormz'])
                peaks = clean_spectrum(peaks=spec['peaks'], max_mz=precursor_mz-1.6, noise_threshold=0.01, ms2_da=0.05)
                peaks = neutral_loss_transform(precursor_mz, peaks)
            except:
                continue

            if len(peaks) > 0:
                if ion_mode not in all_spectra:
                    all_spectra[ion_mode] = []
                # Record the spectrum.
                all_spectra[ion_mode].append({'peaks': peaks,
                                              'precursor_mz': precursor_mz,
                                              'id': spec['db#']})

    print(f'Loaded {len(all_spectra["P"])} positive spectra and {len(all_spectra["N"])} negative spectra.')
    return all_spectra


if __name__ == '__main__':
    main()
