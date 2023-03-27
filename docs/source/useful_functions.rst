================
Useful functions
================

Get the top-n results
=====================

After searching the spectral library, you may want to see only the top-n results or the results with a similarity score higher than a certain threshold. To achieve this, you can use the ``get_topn_matches`` function.

This function takes three parameters:

- ``similarity_array``: The similarity scores returned by the search function.

- ``topn``: The number of top results you want to get. Set to None to get all results.

- ``min_similarity``: The minimum similarity score you want to get. Set to None to get all results.

The function returns a list of dictionaries, where each dictionary represents a spectrum in the library. The dictionary is the same as the one in the library spectra (the input of ``set_library_spectra``), except that the ``peaks`` key is removed and the ``entropy_similarity`` key is added to store the similarity score of the spectrum.

.. code-block:: python

    topn_match = entropy_search.get_topn_matches(entropy_similarity, topn=3, min_similarity=0.01)


This example will return a list of the top 3 matches with a similarity score greater than 0.01.

Save and load index
===================

You can use the python's built-in ``pickle`` module to save and load the ``FlashEntropySearch`` object, like this:

.. code-block:: python

    import pickle
    # Save the index
    with open('path/to/index', 'wb') as f:
        pickle.dump(entropy_search, f)
    # And load the index
    with open('path/to/index', 'rb') as f:
        entropy_search = pickle.load(f)

Meanwhile, we also provide ``read`` and ``write`` functions to save and load the index.

To write ``FlashEntropySearch`` object into disk:

.. code-block:: python

    entropy_search.write('path/to/index')


To read ``FlashEntropySearch`` object from disk:

.. code-block:: python

    entropy_search = FlashEntropySearch()
    entropy_search.read('path/to/index')


If you have a very large spectral library, or your computer's memory is limited, you can use the `low_memory` parameter to partially load the library and reduce the memory usage. For exmaple:

.. code-block:: python

    entropy_search = FlashEntropySearch(low_memory=True)
    entropy_search.read('path/to/index')


The index only needs to be built once. After that, you can use the read function to load the index. If you built the index using the low_memory=False mode, you can still load it using the FlashEntropySearch object with either the low_memory=False or low_memory=True mode.
