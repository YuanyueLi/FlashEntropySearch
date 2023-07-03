[![DOI](https://zenodo.org/badge/612393621.svg)](https://zenodo.org/badge/latestdoi/612393621)

# Flash Entropy Search

This repository contains the original source code, benchmark data, and figures for the manuscript: 

> Li, Y., Fiehn, O., Flash entropy search to query all mass spectral libraries in real time. 04 April 2023, PREPRINT (Version 1) available at Research Square. [https://doi.org/10.21203/rs.3.rs-2693233/v1](https://doi.org/10.21203/rs.3.rs-2693233/v1)

To utilize the Flash Entropy Search algorithm, we offer a standalone software with a Graphical User Interface (GUI) named Entropy Search. This software can be downloaded from our GitHub page: [Entropy Search GitHub Release Page (https://github.com/YuanyueLi/EntropySearch/releases)](https://github.com/YuanyueLi/EntropySearch/releases). EntropySearch is compatible with Linux, Mac, and Windows operating systems.

If you want to incorporate the Flash Entropy Search algorithm into your own code, we provide a Python implementation of the algorithm in the `MSEntropy` repository, which can be found here: [MSEntropy GitHub Repository (https://github.com/YuanyueLi/MSEntropy)](https://github.com/YuanyueLi/MSEntropy).

For comprehensive details on the `Flash Entropy Search` and `MSEntropy` package, refer to our latest documentation: [MSEntropy Documentation (https://msentropy.readthedocs.io/)](https://msentropy.readthedocs.io/).

------------------------------------------------------------------------

## Screenshots for the Entropy Search software

![Screenshot of GUI Input page](./docs/images/GUI_start.png)
![Screenshot of GUI Result Display](./docs/images/GUI_result.png)

------------------------------------------------------------------------

## Incorporate the Flash Entropy Search Algorithm in Your Code

To use the Flash Entropy Search algorithm in your own project, please refer to our MSEntropy package: [MSEntropy GitHub Repository](https://github.com/YuanyueLi/MSEntropy).

### Installation

```bash
pip install ms_entropy
```

### In brief

```python
from ms_entropy import FlashEntropySearch
entropy_search = FlashEntropySearch()

# Step 1: Build the index from the library spectra
entropy_search.build_index(spectral_library)

# Step 2: Search the library
entropy_similarity = entropy_search.search(
    precursor_mz = query_spectrum_precursor_mz, 
    peaks = query_spectrum_peaks
)
```

### In details

Suppose you have a spectral library, you need to format it like this:

```python
import numpy as np
spectral_library = [{
    "id": "Demo spectrum 1",
    "precursor_mz": 150.0,
    "peaks": [[100.0, 1.0], [101.0, 1.0], [103.0, 1.0]]
}, {
    "id": "Demo spectrum 2",
    "precursor_mz": 200.0,
    "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32),
    "metadata": "ABC"
}, {
    "id": "Demo spectrum 3",
    "precursor_mz": 250.0,
    "peaks": np.array([[200.0, 1.0], [101.0, 1.0], [202.0, 1.0]], dtype=np.float32),
    "XXX": "YYY",
}, {
    "precursor_mz": 350.0,
    "peaks": [[100.0, 1.0], [101.0, 1.0], [302.0, 1.0]]}]
```

Note that the `precursor_mz` and `peaks` keys are required, the reset of the keys are optional.

Then you have your query spectrum looks like this:

```python
query_spectrum = {"precursor_mz": 150.0,
                  "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)}
```

You can call the `FlashEntropySearch` class to search the library like this:

```python
from ms_entropy import FlashEntropySearch
entropy_search = FlashEntropySearch()
# Step 1: Build the index from the library spectra
spectral_library = entropy_search.build_index(spectral_library)
# Step 2: Search the library
entropy_similarity = entropy_search.search(
    precursor_mz=query_spectrum['precursor_mz'], peaks=query_spectrum['peaks'])
```

After that, you can print the results like this:

```python
import pprint
pprint.pprint(entropy_similarity)
```

The result will look like this:

```python
{'hybrid_search': array([0.6666666 , 0.99999994, 0.99999994, 0.99999994], dtype=float32),
 'identity_search': array([0.6666667, 0.       , 0.       , 0.       ], dtype=float32),
 'neutral_loss_search': array([0.6666666, 0.       , 0.6666666, 0.3333333], dtype=float32),
 'open_search': array([0.6666666 , 0.99999994, 0.3333333 , 0.6666666 ], dtype=float32)}
```

------------------------------------------------------------------------

## Ways to Calculate Spectral Entropy and Entropy Similarity

There are several ways you can calculate spectral entropy and entropy similarity, either through our GUI or by integrating our package into your code.

### Using the GUI

Our GUI provides a user-friendly way to visualize and calculate entropy similarity:

- For a straightforward approach to real-time visualize and calculate entropy similarity for two MS/MS spectra, use the [MS Viewer web app](https://yuanyueli.github.io/MSViewer).

- To search one spectral file against another spectral file or a spectral library, use the [Entropy Search GUI](https://github.com/YuanyueLi/EntropySearch). The GUI supports `.mgf`, `.msp`, `.mzML`, and `.lbm2` file formats.

### Coding with Our Package

If you prefer to integrate our tools directly into your code, visit the [MSEntropy repository](https://github.com/YuanyueLi/MSEntropy) for the latest version of our code.

- To calculate spectral entropy or entropy similarity:

  - **Python** users: use the [`ms-entropy` package](https://pypi.org/project/ms-entropy/). Find the documentation [here](https://msentropy.readthedocs.io/).

  - **R** users: use the [`msentropy` package](https://cran.r-project.org/web/packages/msentropy/index.html). Documentation is available [here](https://cran.r-project.org/web/packages/msentropy/msentropy.pdf).

  - **C/C++** users: refer to the examples in the [languages/c folder of `MSEntropy` repository](https://github.com/YuanyueLi/MSEntropy/tree/main/languages/c).

  - **JavaScript** users: refer to the examples in the [languages/javascript folder of `MSEntropy` repository](https://github.com/YuanyueLi/MSEntropy/tree/main/languages/javascript).

- To use the Flash entropy search algorithm to search a spectral file against a large spectral library:

  Currently, the Flash entropy search algorithm is only available in **Python**. Use the [`ms-entropy` package](https://pypi.org/project/ms-entropy/). Find the documentation [here](https://msentropy.readthedocs.io/).
