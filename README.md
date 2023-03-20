# The source code for Flash Entropy Search

You can find the benchmark result and the original code used in our manuscript under the `manuscript` folder. We improved our code for easier use, the latest version of the code can be found under the `ms_entropy` folder.

# Installation

## Requirements

- Python >= 3.8
- numpy >= 1.18
- cython >= 0.29
- cupy >= 8.3.0 (optional, only required for GPU acceleration)

The `numpy` and `cython` dependencies will be installed automatically when you install the package from PyPI. The `cupy` dependency is optional, which is only required for GPU acceleration. You need to install `cupy` manually if you want to use GPU acceleration.

## Install

### From PyPI

```bash
pip install ms_entropy
```

- When installing in Windows, you may need to install the [Microsoft Visual C++ 14.0 or greater Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.

### From source

```bash
git@github.com:YuanyueLi/FlashEntropySearch.git
cd FlashEntropySearch
python setup.py build_ext
```

## Test

```bash
python example.py
```

# Usage

## In brief

- Please note the code below is just a brief example, which can not be run directly. For more details, please see the next section (In detail) below.

1. Step 1: Build index

    ```python
    from ms_entropy import FlashEntropySearch
    flash_entropy = FlashEntropySearch()
    flash_entropy.build_index(spectral_library)
    ```

2. Step 2: Clean the query spectrum.

    ```python
    query_peaks = flash_entropy.clean_spectrum_for_search(...)
    ```

   or

    ```python
    from ms_entropy import clean_spectrum
    query_peaks = clean_spectrum(...)
    ```

3. Step 3: Search the library

    ```python
    similarity = flash_entropy.identity_search(...) # Identity search
    similarity = flash_entropy.open_search(...) # Open search
    similarity = flash_entropy.neutral_loss_search(...) # Neutral loss search
    similarity = flash_entropy.hybrid_search(...) # Hybrid search
    ```

4. Step 4: Get the top-n results (optional)

    ```python
    top_n = flash_entropy.get_topn_matches(similarity, topn=..., min_similarity=...)
    ```

## In detail

### Step 0: Prepare the library spectra

The library spectra needs to be represented as a list of dictionaries, where each dictionary represents a spectrum. The following example shows the format of the library spectra.

In the dictionary, the key "precursor_mz" and "peaks" are required, which represent the precursor m/z and the peaks of the spectrum, respectively. The "precursor_mz" needs to be a float number, while the "peaks" can be either a list of lists or a numpy array, which in format of [[mz, intensity], ...]. The other keys are optional, which can be used to store the metadata of the spectrum.

```python
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

### Step 1: Build index

This step is to build the index for the library spectra, which is required for the Flash entropy search. Which can be done by calling the `build_index` function of the `FlashEntropySearch` class. The spectra in the `spectral_library` do not need any pre-processing. The index building process will do all the pre-processing automatically.

Please note that for the fast identity search, all spectra in the `spectral_library` will be **re-sorted** by the m/z values. The `build_index` function will return a list of the re-sorted spectra, which can be used to map the results back to the original order. You can also add some metadata like `id` to the spectra to keep track of the original order.

```python
from ms_entropy import FlashEntropySearch
flash_entropy = FlashEntropySearch()
flash_entropy.build_index(spectral_library)
```

### Step 2: Clean the query spectrum

Before library search, your query spectrum needs to be pre-processed by the `clean_spectrum_for_search` function. The `clean_spectrum_for_search` function will do the following things:

1. The empty peaks (m/z = 0 or intensity = 0) will be removed.

2. Remove the peaks have m/z lower than the 0.

3. Remove the peaks have m/z higher than the `precursor_mz - precursor_ions_removal_da`. This step can be used for remove precursor ions, which can improve the spectral comparison quality.

4. Centroid the spectrum by merging the peaks within the +/- `min_ms2_difference_in_da`, sort the result spectrum by m/z.

5. Remove the peaks with intensity less than the `noise_threshold` \* maximum (intensity).

6. Keep the top `max_peak_num` peaks, and remove the rest peaks.

7. Normalize the intensity to sum to 1.

```python
query_spectrum['peaks'] = flash_entropy.clean_spectrum_for_search(precursor_mz=query_spectrum['precursor_mz'],peaks=query_spectrum['peaks'])
```

We also provide a function `clean_spectrum` to do the same thing, which can be called as follows:

```python
from ms_entropy import clean_spectrum
precursor_ions_removal_da=1.6
query_spectrum['peaks'] = clean_spectrum(spectum=query_spectrum['peaks'], max_mz=query_spectrum['precursor_mz']-precursor_ions_removal_da)
```

These two functions do the same thing, you can choose either one.

### Step 3: Search the library

We provide the following four search functions for the library search:

- `identity_search` --> Identity search

- `open_search` --> Open search

- `neutral_loss_search` --> Neutral loss search

- `hybrid_search` --> Hybrid search

The four functions have those parameters: `precursor_mz`, `peaks`, `ms1_tolerance_in_da`, `ms2_tolerance_in_da`, `target`, and return the similarity score of each spectrum in the library, in the same order as the spectral library returned by the `build_index` function.

The example below shows how to use those functions to search the library.

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

After you search spectral library, most spectral similarity will be 0. And you may only want to see the top-n results or the results with similarity higher than a threshold. You can use the `get_topn_matches` function to get the top-n results.

This function accepts three parameters:
`similarity_array`: the similarity score returned by the search function.
`topn`: the number of top results you want to get, set to `None` to get all results.
`min_similarity`: the minimum similarity score you want to get, set to `None` to get all results.

This function will return a list of dictionaries, every dictionary represents a spectrum in the library. The dictionary is the same as the dictionary in the library spectra (the input of `build_index`), except that the `peaks` key is removed. The `entropy_similarity` key is added to store the similarity score of the spectrum.

```python
topn_match = flash_entropy.get_topn_matches(entropy_similarity, topn=3, min_similarity=0.01)
```

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

If you have a super large spectral library, or your computer's memory is low, you can use the `low_memory` parameter to partially load the spectral library. For exmaple:

```python
flash_entropy = FlashEntropySearch(low_memory=True)
flash_entropy.read('path/to/index')
```

The index only need to be build for one time. After that, you can use the `read` function to load the index. The index built by the `low_memory=Flase` mode can be loaded by the `FlashEntropySearch` object with both `low_memory=Flase` and `low_memory=True` mode.

# Example

We provide some examples for you to better understand how to use the package. You can find the examples in the root directory of the package.

- `example.py` --> An example shows how to use the Flash entropy search from scratch.
- `example_search_mona_method_1.py` --> An example shows how to use the Flash entropy search to search the [MassBank.us (MoNA)](https://massbank.us/) database.
- `example_search_mona_method_2_low_memory.py` --> An example shows how to use the Flash entropy search to search the [MassBank.us (MoNA)](https://massbank.us/) database. This example uses less memory than the `example_search_mona_method_1.py` example.
- `example_search_mona_method_3.py` --> An other example shows how to use the Flash entropy search to search the [MassBank.us (MoNA)](https://massbank.us/) database.

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
