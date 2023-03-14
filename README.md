# FlashEntropySearch

This is a Python implementation of the Flash Entropy Search algorithm. If you have any questions, feel free to contact me.

## Installation

- The program is been tested on Kde Neon 5.26 (Ubuntu 22.04).

### Requirements

- Python 3.9
- Numpy
- Cupy (optional, required for GPU support)

### Build

- The script needs to be built before use.

```bash
python setup.py build_ext --inplace
```

## Use Flash Entropy Search with example

The following example scripts shows how to download all MassBank.us(MoNA) spectra, build an index for the spectra, and perform the Flash Entropy Search (identity, open, neutral loss, and hybrid search) of a given mass spectrum.

Please note that the example scripts will download all MassBank.us(MoNA) data, and extract all spectra from the downloaded data. Which might need about 3-4 GB of disk space. Meanwhile, parsing all spectra will take about 10-30 minutes depending on the computer. The library index will take a few more minutes to build. Downloading database, extracting spectra, and building the index will only be performed once. The index will be saved in the `data` directory. The index will be loaded automatically when the example scripts are run again.

Perform Flash Entropy Search on a given mass spectrum usually takes **_less than 10 milliseconds_**. And the top 5 results will be shown. You can modify the example scripts by yourself to search more spectra or to show more results.

- This example script will download all MassBank.us(MoNA) data, build an index, and perform the identity and open search of a given mass spectrum.

```bash
python example_identity_open_search.py
```

- This example script will download all MassBank.us(MoNA) data, build an index, and perform the neutral loss search of a given mass spectrum.

```bash
python example_neutral_loss_search.py
```

- This example script will download all MassBank.us(MoNA) data, build an index, and perform the hybrid search of a given mass spectrum.

```bash
python example_hybrid_search.py
```

## Use Flash Entropy Search with your own code

To integrate the Flash Entropy Search library into your own code, you need to do the following steps:

1. Clean the spectra.
   We provide a function `clean_spectrum` to clean the spectra, using the following parameters:

   - peaks: The peaks of the spectrum, a numpy array with shape (n, 2), where n is the number of peaks, and the first column is the m/z values, and the second column is the intensity values. This numpy array needs to be in the dtype `np.float32`.
   - min_mz: The minimum m/z value to keep. Default is 0.0.
   - max_mz: The maximum m/z value to keep. Default is -1, which will not remove any peaks. Set it to the precursor_mz - 1.6 for MS2 spectra is highly recommended.
   - max_peak_num: The maximum number of peaks to keep. Default is -1, which will keep all peaks.
   - noise_threshold: The noise threshold to remove the low intensity peaks. Default is 0.01.
   - normalize_intensity: Whether to normalize the intensity values. Default is 1, which will normalize the sum of the intensity values to 1.0. Set it to 1 is required for the Flash Entropy Search.
   - ms2_da: The m/z tolerance for MS2 spectra. Need to be at least twice the m/z tolerance of the search. Default is 0.05.

2. Build the library index.

   This step can be done with the following code:

   ```python
   flash_entropy = EntropySearch()
   spectral_library = flash_entropy.build_index(spectral_library,sort_by_precursor_mz=True)
   ```

   - Please note that the `spectral_library` will be re-sorted for faster identity search. If you don't want to sort the library, you can set `sort_by_precursor_mz` to `False`, but you can't perform identity search.

   The `spectral_library` is a list of dictionaries, each dictionary contains the information of a spectrum. The following keys are required:

   - precursor_mz: The precursor m/z value of the spectrum.
   - peaks: The peaks of the spectrum, need to the output of the `clean_spectrum` function. A numpy array with shape (n, 2), where n is the number of peaks, and the first column is the m/z values, and the second column is the intensity values. This numpy array needs to be in the dtype `np.float32`.

3. Perform the search.

   This step can be done with the following code:

   ```python
    entropy_similarity_for_identity_search = flash_entropy.search_identity(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
        ms1_tolerance_in_da=0.01,
        ms2_tolerance_in_da=0.02)

    entropy_similarity_for_open_search = flash_entropy.search_open(
        peaks=query_spectrum['peaks'],
        ms2_tolerance_in_da=0.02)
   ```
    - The `search_identity` function will perform the identity search, and return the entropy similarity values for each spectrum in the library.
    - The `search_open` function will perform the open search, and return the entropy similarity values for each spectrum in the library.

    - The peaks of the query spectrum need to be the output of the `clean_spectrum` function.
    

The following example code shows how to use the Flash Entropy Search library in your own code.

```python
import numpy as np
from flash_entropy.entropy_search import EntropySearch
from flash_entropy.functions import clean_spectrum

# This is your library spectra, here the "precursor_mz" and "peaks" are required. The "id" is optional.
spectral_library = [
    {
        "id": "Demo spectrum 1",
        "precursor_mz": 150.0,
        "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [103.0, 1.0]], dtype=np.float32)
    },
    {
        "id": "Demo spectrum 2",
        "precursor_mz": 250.0,
        "peaks": np.array([[200.0, 1.0], [201.0, 1.0], [202.0, 1.0]], dtype=np.float32)
    },
    {
        "id": "Demo spectrum 3",
        "precursor_mz": 200.0,
        "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)
    }
]
# This is your query spectrum, here the "peaks" is required, the "precursor_mz" is required for identity search.
query_spectrum = {
    "precursor_mz": 150.0,
    "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)
}

#################### Step 1: Clean the spectra. ####################
# Clean the library spectra.
for spectrum in spectral_library:
    spectrum["peaks"] = clean_spectrum(peaks=spectrum['peaks'], max_mz=spectrum['precursor_mz']-1.6, noise_threshold=0.01, ms2_da=0.05)

# Clean the query spectrum.
query_spectrum["peaks"] = clean_spectrum(peaks=query_spectrum['peaks'], max_mz=query_spectrum['precursor_mz']-1.6, noise_threshold=0.01, ms2_da=0.05)

#################### Step 2: Build the library index. ####################
flash_entropy = EntropySearch()
# Please note that the library spectra will be re-sorted by precursor_mz for fast identity search.
spectral_library = flash_entropy.build_index(spectral_library, sort_by_precursor_mz=True)
# This step is optional, just used to show the spectrum id in the search results.
flash_entropy.library_id = [spectrum['id'] for spectrum in spectral_library]

#################### Step 3: Perform the Flash entropy search. ####################
# Perform the identity search.
entropy_similarity = flash_entropy.search_identity(precursor_mz=query_spectrum['precursor_mz'],
                                                   peaks=query_spectrum['peaks'],
                                                   ms1_tolerance_in_da=0.01, ms2_tolerance_in_da=0.02)
# Output the best match.
best_match = np.argmax(entropy_similarity)
print(f"Best identity search match: {flash_entropy.library_id[best_match]}, entropy similarity: {entropy_similarity[best_match]:.4f}")

# Perform the open search.
entropy_similarity = flash_entropy.search_open(peaks=query_spectrum['peaks'], ms2_tolerance_in_da=0.02)

# Output the best match.
best_match = np.argmax(entropy_similarity)
print(f"Best open search match: {flash_entropy.library_id[best_match]}, entropy similarity: {entropy_similarity[best_match]:.4f}")
```

The code will output the following results:

```bash
Best identity search match: Demo spectrum 1, entropy similarity: 0.6667
Best open search match: Demo spectrum 3, entropy similarity: 1.0000
```
