# The source code for Flash Entropy Search

This repository contains the source code for Flash Entropy Search, a method using entropy similarity for fast searching mass spectrometry spectral library.

You can find the benchmark results and the original code used in our manuscript under the `manuscript` folder.

We are continuously improving the code, and the latest version of the code can be found under the `ms_entropy` folder.

The API documentation for the latest version can be found [here](https://flashentropysearch.readthedocs.io/en/develop/).

# Installation

## Requirements

To use this package, you need to have the following software installed on your system:

- Python >= 3.8

- C compiler and Python development headers.

  - When installing on Linux, you may need to install the `gcc` and `python-dev` packages first.
  - When installing on Windows, you may need to install the [Microsoft Visual C++ 14.0 or greater Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.

- numpy >= 1.18 (will be installed automatically when you install the package from PyPI)

- Cython >= 0.29 (will be installed automatically when you install the package from PyPI)

- cupy >= 8.3.0 (optional, only required for GPU acceleration)

The `numpy` and `cython` dependencies will be installed automatically when you install the package from PyPI. The `cupy` dependency is optional, and is only required for GPU acceleration. If you want to use GPU acceleration, you need to install `cupy` manuall before installing the package from PyPI.

The code is tested on KDE neon 5.27, but it should work on other platforms as well.

## Install

### From PyPI

To install the latest version of the package from PyPI, run the following command:

```bash
pip install ms_entropy
```

- The installation time should be less than 1 minute, if everything is set up correctly.

### From source

To install from source, clone the repository and run the following commands:

```bash
git@github.com:YuanyueLi/FlashEntropySearch.git
cd FlashEntropySearch
python setup.py build_ext
```

## Test

To test that the package is working correctly, run the example.py script:

```bash
python example.py
```

# Usage

- You can find an example of how to use the package in the `example.py` file.

## In brief

- Please note the code below is just a brief example, which can not be run directly. For more details, please see the next section (In detail) below.

  ### Step 1: Build index

  First, you need to build an index of the spectral library you want to search. You can do this by creating an instance of the `FlashEntropySearch` class and calling the `build_index` method with your spectral library data:

  ```python
  from ms_entropy import FlashEntropySearch
  flash_entropy = FlashEntropySearch()
  flash_entropy.build_index(spectral_library)
  ```

  ### Step 2: Clean the query spectrum.

  Before searching the library, you need to clean the query spectrum using the `clean_spectrum` function:

  ```python
  query_peaks = flash_entropy.clean_spectrum_for_search(...)
  ```

  Alternatively, you can use the clean_spectrum_for_search method of the FlashEntropySearch class to do the same thing:

  ```python
  from ms_entropy import clean_spectrum
  query_peaks = clean_spectrum(...)
  ```

  ### Step 3: Search the library

  Once you have built the index and cleaned the query spectrum, you can perform various types of searches on the spectral library using the FlashEntropySearch class methods:

  ```python
  similarity = flash_entropy.identity_search(...) # Identity search
  similarity = flash_entropy.open_search(...) # Open search
  similarity = flash_entropy.neutral_loss_search(...) # Neutral loss search
  similarity = flash_entropy.hybrid_search(...) # Hybrid search
  ```

  ### Step 4: Get the top-n results (optional)

  You can use the `get_topn_matches` method of the `FlashEntropySearch` class to get the top n matches from the similarity scores:

  ```python
  top_n = flash_entropy.get_topn_matches(similarity, topn=..., min_similarity=...)
  ```

## In detail

### Step 0: Prepare the library spectra

The library spectra needs to be represented as a list of dictionaries, where each dictionary represents a spectrum. The following example shows the format of the library spectra.

```python
import numpy as np
spectral_library = [
    {
        "precursor_mz": 150.0,
        "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [103.0, 1.0]], dtype=np.float32)
    },
    {
        "id": "Demo spectrum 2",
        "precursor_mz": 250.0,
        "peaks": np.array([[200.0, 1.0], [201.0, 1.0], [202.0, 1.0]], dtype=np.float32)
    },
    {
        "metadata": "ABC",
        "XXX": "YYY",
        "precursor_mz": 200.0,
        "peaks":[[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]]
    }
]
```

In the dictionary, the key "precursor_mz" and "peaks" are required, which represent the precursor m/z and the peaks of the spectrum, respectively. The "precursor_mz" needs to be a float number, while the "peaks" can be either a list of lists or a numpy array, which in format of [[mz, intensity], ...]. The other keys are optional, which can be used to store the metadata of the spectrum.

### Step 1: Build index

The next step is to build an index for the library spectra, which is required for the Flash entropy search. To build the index, call the `build_index` function of the `FlashEntropySearch` class, passing in the library spectra as an argument. The spectra in the `spectral_library` do not require any pre-processing, as the index building process will handle this automatically.

```python
from ms_entropy import FlashEntropySearch
flash_entropy = FlashEntropySearch()
flash_entropy.build_index(spectral_library)
```

It is important to note that for the fast identity search, all spectra in the `spectral_library` will be **re-sorted** by the m/z values. The `build_index` function will return a list of the re-sorted spectra, which can be used to map the results back to the original order. If necessary, you can also add some metadata, such as an `id` field, to the spectra to keep track of their original order.

### Step 2: Clean the query spectrum

Before performing a library search, the query spectrum needs to be pre-processed using the `clean_spectrum_for_search` function. This function performs the following steps:

1. Remove empty peaks (m/z <= 0 or intensity <= 0).

2. Remove peaks with m/z values greater than `precursor_mz - precursor_ions_removal_da`. This step removes precursor ions, which can improve the quality of spectral comparison.

3. Centroid the spectrum by merging peaks within +/- `min_ms2_difference_in_da` and sort the resulting spectrum by m/z.

4. Remove peaks with intensity less than `noise_threshold` \* maximum intensity.

5. Keep only the top `max_peak_num` peaks and remove all others.

6. Normalize the intensity to sum to 1.

To use this function, call it on your query spectrum, passing in the relevant parameters:

```python
query_spectrum['peaks'] = flash_entropy.clean_spectrum_for_search(precursor_mz=query_spectrum['precursor_mz'],peaks=query_spectrum['peaks'])
```

We also provide a separate function called `clean_spectrum` that performs the same cleaning steps as `clean_spectrum_for_search`. This function can be called as follows:

```python
from ms_entropy import clean_spectrum
precursor_ions_removal_da=1.6
query_spectrum['peaks'] = clean_spectrum(spectum=query_spectrum['peaks'], max_mz=query_spectrum['precursor_mz']-precursor_ions_removal_da)
```

These two functions do the same thing and can be used interchangeably, you can choose either one.

### Step 3: Search the library

We provide four search functions for library search:

- `identity_search` --> Identity search

- `open_search` --> Open search

- `neutral_loss_search` --> Neutral loss search

- `hybrid_search` --> Hybrid search

Each function takes the query spectrum as input, along with the spectral library index built in Step 1, and returns the similarity score of each spectrum in the library, in the same order as the spectral library returned by the `build_index` function.

- Here are the parameters that each search function takes:

  `precursor_mz`: The precursor m/z of the query spectrum.

  `peaks`: The peaks of the query spectrum.

  `ms1_tolerance_in_da`: The mass tolerance to use for the precursor m/z in Da.

  `ms2_tolerance_in_da`: The mass tolerance to use for the fragment ions in Da.

  `target`: Run the similarity calculation on cpu or gpu. The default value is "cpu".

Here's an example of how to use these functions:

```python
# Identity search
entropy_similarity = flash_entropy.identity_search(precursor_mz=query_spectrum['precursor_mz'],
                                                   peaks=query_spectrum['peaks'],
                                                   ms1_tolerance_in_da=0.01, ms2_tolerance_in_da=0.02)

# Open search
entropy_similarity = flash_entropy.open_search(peaks=query_spectrum['peaks'], ms2_tolerance_in_da=0.02)

# Neutral loss search
entropy_similarity = flash_entropy.neutral_loss_search(precursor_mz=query_spectrum['precursor_mz'],
                                                       peaks=query_spectrum['peaks'],
                                                       ms2_tolerance_in_da=0.02)

# Hybrid search
entropy_similarity = flash_entropy.hybrid_search(precursor_mz=query_spectrum['precursor_mz'],
                                                 peaks=query_spectrum['peaks'],
                                                 ms2_tolerance_in_da=0.02)

```

### Step 4: Get the top-n results (optional)

After searching the spectral library, you may want to see only the top-n results or the results with a similarity score higher than a certain threshold. To achieve this, you can use the `get_topn_matches` function.

This function takes three parameters:

`similarity_array`: The similarity scores returned by the search function.

`topn`: The number of top results you want to get. Set to None to get all results.

`min_similarity`: The minimum similarity score you want to get. Set to None to get all results.

The function returns a list of dictionaries, where each dictionary represents a spectrum in the library. The dictionary is the same as the one in the library spectra (the input of `build_index`), except that the `peaks` key is removed and the `entropy_similarity` key is added to store the similarity score of the spectrum.

```python
topn_match = flash_entropy.get_topn_matches(entropy_similarity, topn=3, min_similarity=0.01)
```

This example will return a list of the top 3 matches with a similarity score greater than 0.01.

## Misc: save and load index

You can use the python's built-in `pickle` module to save and load the `FlashEntropySearch` object, like this:

```python
import pickle
# Save the index
with open('path/to/index', 'wb') as f:
    pickle.dump(flash_entropy, f)
# And load the index
with open('path/to/index', 'rb') as f:
    flash_entropy = pickle.load(f)
```

Meanwhile, we also provide `read` and `write` functions to save and load the index.

To write `FlashEntropySearch` object into disk:

```python
flash_entropy.write('path/to/index')
```

To read `FlashEntropySearch` object from disk:

```python
flash_entropy = FlashEntropySearch()
flash_entropy.read('path/to/index')
```

If you have a very large spectral library, or your computer's memory is limited, you can use the `low_memory` parameter to partially load the library and reduce the memory usage. For exmaple:

```python
flash_entropy = FlashEntropySearch(low_memory=True)
flash_entropy.read('path/to/index')
```

The index only needs to be built once. After that, you can use the read function to load the index. If you built the index using the low_memory=False mode, you can still load it using the FlashEntropySearch object with either the low_memory=False or low_memory=True mode.

# Example

We have included several examples in the root directory of the package to help you better understand how to use it. These examples cover a range of use cases and demonstrate how to perform common tasks such as building an index, searching for spectra, and evaluating search performance.

## `example.py`

An example shows how to use the Flash entropy search from scratch. The running time should be less than 1 second, and the expected output should be:

```text
    -------------------- Identity search --------------------
    [{'entropy_similarity': 0.6666667,
    'id': 'Demo spectrum 1',
    'precursor_mz': 150.0}]
    -------------------- Open search --------------------
    [{'entropy_similarity': 0.99999994,
    'id': 'Demo spectrum 3',
    'precursor_mz': 200.0},
    {'entropy_similarity': 0.6666666,
    'id': 'Demo spectrum 4',
    'precursor_mz': 350.0},
    {'entropy_similarity': 0.6666666,
    'id': 'Demo spectrum 1',
    'precursor_mz': 150.0}]
    -------------------- Neutral loss search --------------------
    [{'entropy_similarity': 0.6666666,
    'id': 'Demo spectrum 2',
    'precursor_mz': 250.0},
    {'entropy_similarity': 0.6666666,
    'id': 'Demo spectrum 1',
    'precursor_mz': 150.0},
    {'entropy_similarity': 0.3333333,
    'id': 'Demo spectrum 4',
    'precursor_mz': 350.0}]
    -------------------- Hybrid search --------------------
    [{'entropy_similarity': 0.99999994,
    'id': 'Demo spectrum 4',
    'precursor_mz': 350.0},
    {'entropy_similarity': 0.99999994,
    'id': 'Demo spectrum 2',
    'precursor_mz': 250.0},
    {'entropy_similarity': 0.99999994,
    'id': 'Demo spectrum 3',
    'precursor_mz': 200.0}]
```

## `example_search_mona_method_1.py`

An example shows how to use the Flash entropy search to search the whole [MassBank.us (MoNA)](https://massbank.us/) database.

The first time you run this example, it will take about 10-20 minutes to download the spectra from MoNA and parse the spectra from .msp files. And it will take about 2-4 minutes to build the index for MoNA library. The second time you run this example, the index will be loaded directly from disk. The running time should be less than 1 second, and the expected output should be:

```text
    Loading index
    Downloading https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/03d5a22c-c1e1-4101-ac70-9a4eae437ef5 to /p/FastEntropySearch/github_test/data/mona-2023-03-23.zip
    Loading spectra from /p/FastEntropySearch/github_test/data/mona-2023-03-23.zip, this may take a while.
    Loaded 811840 positive spectra and 1198329 negative spectra.
    Building index, this will only need to be done once.
    Building index for spectra with ion mode P
    Building index for spectra with ion mode N
    Saving index
    ********************************************************************************
    Identity Search with Flash Entropy Search
    Finished identity search in 0.0017 seconds with 1196680 results.
    Top 5 matches:
    Rank 1: AU116754 with score 1.0000
    Rank 2: AU116755 with score 0.8081
    Rank 3: AU116753 with score 0.6565
    Rank 4: AU116752 with score 0.2717
    ********************************************************************************
    Open Search with Flash Entropy Search
    Finished open search in 0.0006 seconds with 1196680 results.
    Top 5 matches:
    Rank 1: AU116754 with score 1.0000
    Rank 2: AU116755 with score 0.8081
    Rank 3: AU116753 with score 0.6565
    Rank 4: CCMSLIB00004751228 with score 0.4741
    Rank 5: LU040151 with score 0.4317
    ********************************************************************************
    Neutral Loss Search with Flash Entropy Search
    Finished neutral loss search in 0.0006 seconds with 1196680 results.
    Top 5 matches:
    Rank 1: AU116754 with score 1.0000
    Rank 2: AU116755 with score 0.8081
    Rank 3: AU116753 with score 0.6565
    Rank 4: LipidBlast2022_1230911 with score 0.3796
    Rank 5: LipidBlast2022_1230977 with score 0.3796
    ********************************************************************************
    Hybrid Search with Flash Entropy Search
    Finished hybrid search in 0.0010 seconds with 1196680 results.
    Top 5 matches:
    Rank 1: AU116754 with score 1.0000
    Rank 2: AU116755 with score 0.8081
    Rank 3: AU116753 with score 0.6565
    Rank 4: CCMSLIB00004751228 with score 0.4741
    Rank 5: LU040151 with score 0.4317
```

## `example_search_mona_method_2_low_memory.py`

An example shows how to use the Flash entropy search to search the [MassBank.us (MoNA)](https://massbank.us/) database. This example uses less memory than the `example_search_mona_method_1.py` example.

## `example_search_mona_method_3.py`

An other example shows how to use the Flash entropy search to search the [MassBank.us (MoNA)](https://massbank.us/) database.

# Other features

## Run Flash entropy search on computer with limited memory

First, you need to run `example_search_mona_method_1.py` or `example_search_mona_method_2_low_memory.py` to build the index for MoNA library. The second time you run `example_search_mona_method_1.py` or `example_search_mona_method_2_low_memory.py`, the index will be loaded directly from disk.

After building the index, on my computer, second time run `example_search_mona_method_1.py` takes about 1,212MB memory to search one spectrum against whole MassBank.us (MoNA) library, and second time run `example_search_mona_method_2_low_memory.py` takes about 84MB memory to search one spectrum. This feature is useful when you have a super large spectral library and your computer's memory is limited.

## Run Flash entropy search on GPU

When you have a GPU and searching a single spectrum took you more than 0.1 seconds, you can use the GPU to speed up the search. To use the GPU, you need to install the [cupy](https://cupy.dev/) package first. Then you can use the `target` parameter to `gpu` enable the GPU.

```python
# Identity search
entropy_similarity = flash_entropy.identity_search(target="gpu", ...)
# Open search
entropy_similarity = flash_entropy.open_search(target="gpu", ...)
# Neutral loss search
entropy_similarity = flash_entropy.neutral_loss_search(target="gpu", ...)
# Hybrid search
entropy_similarity = flash_entropy.hybrid_search(target="gpu", ...)
```
