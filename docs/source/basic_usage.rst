====================
Basic usage
====================


Step 0: Prepare the library spectra
===================================

The library spectra needs to be represented as a list of dictionaries, where each dictionary represents a spectrum. The following example shows the format of the library spectra.

.. code-block:: python

    import numpy as np
    spectral_library =  [{
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

In the dictionary, the key ``precursor_mz`` and ``peaks`` are required, which represent the precursor m/z and the peaks of the spectrum, respectively. The ``precursor_mz`` needs to be a float number, while the ``peaks`` can be either a list of lists or a numpy array, which in format of ``[[mz, intensity], [mz, intensity], ...]``. The other keys are optional, which can be used to store the metadata of the spectrum.

Step 1: Build the index
=======================

The first step is to build an index for the library spectra, which is required for the Flash entropy search. To build the index, call the ``build_index`` function of the ``FlashEntropySearch`` class, passing in the library spectra as an argument. The spectra in the ``spectral_library`` do not require any pre-processing, as the index building process will handle this automatically.

.. code-block:: python

    from ms_entropy import FlashEntropySearch
    entropy_search = FlashEntropySearch()
    entropy_search.build_index(spectral_library)


.. warning::
    It is important to note that for the fast identity search, all spectra in the ``spectral_library`` will be ``re-sorted`` by the m/z values. The ``build_index`` function will return a list of the re-sorted spectra, which can be used to map the results back to the original order. If necessary, you can also add some metadata, such as an ``id`` field, to the spectra to keep track of their original order.


Step 2: Search the library
==========================

Next, you can search your query spectrum against the library spectra with the ``search`` function. The ``search`` function takes the query spectrum as input, and returns the similarity score of each spectrum in the library, in the same order as the spectral library returned by the ``build_index`` function.

The ``search`` function takes the following parameters:

- ``precursor_mz``: The precursor m/z of the query spectrum.

- ``peaks``: The peaks of the query spectrum, which can be either a list of lists or a numpy array, which in format of ``[[mz, intensity], [mz, intensity], ...]``.

- ``ms1_tolerance_in_da``: The mass tolerance to use for the precursor m/z in Da, which is only used for the identity search.

- ``ms2_tolerance_in_da``: The mass tolerance to use for the fragment ions in Da.

- ``method``: The search method to use. We provide four search methods: ``identity``, ``open``, ``neutral_loss``, and ``hybrid``. The value can be either a string or a list / set of those four strings like ``{'identity', 'open'}``. You can also use ``all`` to run all four methods. The default value is ``all``.

- ``target``: Run the similarity calculation on cpu or gpu. The default value is ``cpu``.

- ``precursor_ions_removal_da``: The mass tolerance to use for removing the precursor ions in Da. The fragment ions with m/z larger than ``precursor_mz - precursor_ions_removal_da`` will be removed. Based on our experiments, removal of the precursor ions can improve the search performance.

- ``noise_threshold``: The intensity threshold to use for removing the noise peaks. The peaks with intensity smaller than ``noise_threshold * max(fragment ion's intensity)`` will be removed.

- ``max_num_peaks``: Only keep the top ``max_num_peaks`` peaks with the highest intensity. The default value is None, which means no limit.

You can run the search function like this:

.. code-block:: python

    entropy_similarity = entropy_search.search(
        precursor_mz=150.0,
        peaks=[[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]]
    )

The return value of the ``search`` function is a dictionary, where the key is the search method, and the value is a list of similarity scores. The similarity scores are in the same order as the spectral library returned by the ``build_index`` function. The results look like this:

.. code-block:: python

    {
        'identity_search': [0.0, 0.5, 0.0, 0.8],
        'open_search': [0.0, 0.0, 0.3, 0.8],
        'neutral_loss_search': [0.2, 0.0, 0.7, 0.0],
        'hybrid_search': [0.2, 0.5, 1.0, 0.8]
    }



Alternative: search the library using individual search functions
=================================================================

The ``search`` function do two things automatically: (1) clean the query spectrum, and (2) perform the library search. Alternatively, you can also not use the ``search`` function, but do it in two steps manually. First, you can use the ``clean_spectrum_for_search`` function to clean the query spectrum, and then use the individual search functions to search the library. These two ways are equivalent, and you can choose the one that is more convenient for you.

Clean the query spectrum
------------------------

Before performing a library search, the query spectrum needs to be pre-processed using the `clean_spectrum_for_search` function. This function performs the following steps:

1. Remove empty peaks (m/z <= 0 or intensity <= 0).

2. Remove peaks with m/z values greater than ``precursor_mz - precursor_ions_removal_da``. This step removes precursor ions, which can improve the quality of spectral comparison.

3. Centroid the spectrum by merging peaks within +/- ``min_ms2_difference_in_da`` and sort the resulting spectrum by m/z.

4. Remove peaks with intensity less than ``noise_threshold`` * maximum intensity.

5. Keep only the top ``max_peak_num`` peaks and remove all others.

6. Normalize the intensity to sum to 1.

Let's say you have your query spectrum looks like this:

.. code-block:: python

    query_spectrum = {"precursor_mz": 150.0,
                      "peaks": [[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]]}

To use the `clean_spectrum_for_search` function, call it on your query spectrum, passing in the relevant parameters:

.. code-block:: python

    query_spectrum['peaks'] = entropy_search.clean_spectrum_for_search(precursor_mz=query_spectrum['precursor_mz'],peaks=query_spectrum['peaks'])

We also provide a separate function called ``clean_spectrum`` that performs the same cleaning steps as ``clean_spectrum_for_search``. This function can be called as follows:

.. code-block:: python

    from ms_entropy import clean_spectrum
    precursor_ions_removal_da=1.6
    query_spectrum['peaks'] = clean_spectrum(spectum=query_spectrum['peaks'], max_mz=query_spectrum['precursor_mz']-precursor_ions_removal_da)

These two functions do the same thing and can be used interchangeably, you can choose either one.


Search the library using individual search functions
----------------------------------------------------

We provide four search functions for library search:

- ``identity_search`` --> Identity search
- ``open_search`` --> Open search
- ``neutral_loss_search`` --> Neutral loss search
- ``hybrid_search`` --> Hybrid search

Each function takes the ``pre-cleaned`` query spectrum as input, along with the spectral library index built in Step 1, and returns the similarity score of each spectrum in the library, in the same order as the spectral library returned by the ``set_library_spectra`` function.

.. warning::
    When use those four individual search functions, the ``peaks`` needs to be pre-processed by the ``clean_spectrum_for_search`` or ``clean_spectrum`` function. If not, an error will be raised.


Here are the parameters that each search function takes:

- ``precursor_mz``: The precursor m/z of the query spectrum.
- ``peaks``: The peaks of the query spectrum.
- ``ms1_tolerance_in_da``: The mass tolerance to use for the precursor m/z in Da.
- ``ms2_tolerance_in_da``: The mass tolerance to use for the fragment ions in Da.
- ``target``: Run the similarity calculation on cpu or gpu. The default value is "cpu".

Here's an example of how to use these functions:

.. code-block:: python
        
    # Identity search
    entropy_similarity = entropy_search.identity_search(precursor_mz=query_spectrum['precursor_mz'],
                                                    peaks=query_spectrum['peaks'],
                                                    ms1_tolerance_in_da=0.01, ms2_tolerance_in_da=0.02)

    # Open search
    entropy_similarity = entropy_search.open_search(peaks=query_spectrum['peaks'], ms2_tolerance_in_da=0.02)

    # Neutral loss search
    entropy_similarity = entropy_search.neutral_loss_search(precursor_mz=query_spectrum['precursor_mz'],
                                                        peaks=query_spectrum['peaks'],
                                                        ms2_tolerance_in_da=0.02)

    # Hybrid search
    entropy_similarity = entropy_search.hybrid_search(precursor_mz=query_spectrum['precursor_mz'],
                                                    peaks=query_spectrum['peaks'],
                                                    ms2_tolerance_in_da=0.02)
