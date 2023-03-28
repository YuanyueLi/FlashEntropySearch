========
Examples
========

We have included several examples in the root directory of the package to help you better understand how to use it. These examples cover a range of use cases and demonstrate how to perform common tasks such as building an index, searching for spectra, and evaluating search performance.

You can find those examples under the folder ``examples``. The following is a list of examples:

example.py
==========

An example shows how to use the Flash entropy search from scratch. The expected output should be:

.. code-block:: bash

    -------------------- All types of similarity search --------------------
    {'hybrid_search': array([0.6666666 , 0.99999994, 0.99999994, 0.99999994], dtype=float32),
    'identity_search': array([0.6666667, 0.       , 0.       , 0.       ], dtype=float32),
    'neutral_loss_search': array([0.6666666, 0.       , 0.6666666, 0.3333333], dtype=float32),
    'open_search': array([0.6666666 , 0.99999994, 0.3333333 , 0.6666666 ], dtype=float32)}


example-2.py
============

Another example shows how to use the Flash entropy search from scratch. The expected output should be:

.. code-block:: bash

    -------------------- Identity search --------------------
    [{'entropy_similarity': 0.6666667,
    'id': 'Demo spectrum 1',
    'precursor_mz': 150.0}]
    -------------------- Open search --------------------
    [{'entropy_similarity': 0.99999994,
    'id': 'Demo spectrum 2',
    'metadata': 'ABC',
    'precursor_mz': 200.0},
    {'entropy_similarity': 0.6666666, 'precursor_mz': 350.0},
    {'entropy_similarity': 0.6666666,
    'id': 'Demo spectrum 1',
    'precursor_mz': 150.0}]
    -------------------- Neutral loss search --------------------
    [{'XXX': 'YYY',
    'entropy_similarity': 0.6666666,
    'id': 'Demo spectrum 3',
    'precursor_mz': 250.0},
    {'entropy_similarity': 0.6666666,
    'id': 'Demo spectrum 1',
    'precursor_mz': 150.0},
    {'entropy_similarity': 0.3333333, 'precursor_mz': 350.0}]
    -------------------- Hybrid search --------------------
    [{'entropy_similarity': 0.99999994, 'precursor_mz': 350.0},
    {'XXX': 'YYY',
    'entropy_similarity': 0.99999994,
    'id': 'Demo spectrum 3',
    'precursor_mz': 250.0},
    {'entropy_similarity': 0.99999994,
    'id': 'Demo spectrum 2',
    'metadata': 'ABC',
    'precursor_mz': 200.0}]

example-with_individual_search_function.py
==========================================

An example shows how to use the Flash entropy search from scratch, this example is similar to ``example-2.py`` but it uses individual search functions instead of the ``search`` function. The expected output should be the same as the previous example ``example-2.py``.


example-search_mona-with_read_write_functions.py
================================================

An example shows how to use the Flash entropy search to search the whole [MassBank.us (MoNA)](https://massbank.us/) database.

The first time you run this example, it will take about 10-20 minutes to download the spectra from MoNA and parse the spectra from .msp files. And it will take about 2-4 minutes to build the index for MoNA library. The second time you run this example, the index will be loaded directly from disk. The running time should be less than 1 second, and the expected output should be:

.. code-block:: bash

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

example-search_mona-with_read_write_functions-low_memory_usage.py
=================================================================

An example shows how to use the Flash entropy search to search the [MassBank.us (MoNA)](https://massbank.us/) database. This example uses less memory than the ``example-search_mona-with_read_write_functions.py`` example.


First, you need to run `example-search_mona-with_read_write_functions.py` or `example-search_mona-with_read_write_functions-low_memory_usage.py` to build the index for MoNA library. The second time you run `example-search_mona-with_read_write_functions.py` or `example-search_mona-with_read_write_functions-low_memory_usage.py`, the index will be loaded directly from disk.

After building the index, on my computer, second time run `example-search_mona-with_read_write_functions.py` takes about 1,212MB memory to search one spectrum against whole MassBank.us (MoNA) library, and second time run `example-search_mona-with_read_write_functions-low_memory_usage.py` only takes about 84MB memory to search one spectrum. This feature is useful when you have a super large spectral library and your computer's memory is limited.


example-search_mona-with_pickle_functions.py
============================================

An other example shows how to use the Flash entropy search to search the [MassBank.us (MoNA)](https://massbank.us/) database. This example use build-in pickle functions to save and load index.