==============
Advanced usage
==============


Run Flash entropy search with limited memory
============================================

This method is useful when you have a super large spectral library and your computer's memory is limited.

To archive this, you just need to set ``path_data`` parameter to the path of the index file, and set ``low_memory`` parameter to ``True``. Then run the reset of the code as usual. That's it!


.. code-block:: python

    from ms_entropy import FlashEntropySearch

    # Instead of using this:
    # entropy_search = FlashEntropySearch()
    # Use this:
    entropy = FlashEntropySearch(path_data='path/to/library/index', low_memory=True)

    # Then the reset of the code is the same as usual.
    entropy_search.build_index(spectral_library)
    # ...... (the reset of the code is the same as usual)

The index build in normal mode and low memory mode is the same. If you use the our ``write`` and ``read`` method to save and load the index, you can use the index in normal mode and low memory mode interchangeably. For example, you can build the index in normal mode, save it to disk with ``write`` method. After that, you can initialize the ``FlashEntropySearch`` object with ``path_data`` parameter which points to the index file, and set ``low_memory`` parameter to ``True``, then call the ``read`` method to load the index, and do the rest search as usual.


Run Flash entropy search on GPU
===============================

When you have a GPU and searching a single spectrum took you more than 0.1 seconds, you can use the GPU to speed up the search. To use the GPU, you need to install the `Cupy <https://cupy.dev/>`_ package first. Then you can use the `target` parameter to `gpu` enable the GPU.

.. code-block:: python

    from ms_entropy import FlashEntropySearch
    entropy = FlashEntropySearch()
    entropy_search.build_index(spectral_library)

    # Instead of using this:
    # entropy_similarity = entropy_search.search(
    #     precursor_mz=150.0,
    #     peaks=[[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]],
    #     target='cpu'
    # )
    # Use this:
    entropy_similarity = entropy_search.search(
        precursor_mz=150.0,
        peaks=[[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]],
        target='gpu'
    )

    # Then the reset of the code is the same as usual.

The return value of calculating with ``CPU`` and ``GPU`` is the same. You can use the same code to process the result.