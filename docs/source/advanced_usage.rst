==============
Advanced usage
==============


Run Flash entropy search with limited memory
============================================

This method is useful when you have a super large spectral library and your computer's memory is limited.

To archive this, when constructing the ``FlashEntropySearch`` object, you just need to set ``path_data`` parameter to the path of the index file, and set ``low_memory`` parameter to ``True``. Then read the pre-built index file by calling the ``read`` method. After that, all the reset of the code is the same as usual. That's it!

.. code-block:: python

    from ms_entropy import FlashEntropySearch

    # Instead of using this:
    # entropy_search = FlashEntropySearch()
    # Use this:
    entropy_search = FlashEntropySearch(path_data='path/to/library/index', low_memory=True)
    entropy_search.read()

    # Then the reset of the code is the same as usual.
    entropy_search.build_index(spectral_library)
    # ...... (the reset of the code is the same as usual)

The index build in normal mode and low memory mode is the same. If you use the our ``write`` and ``read`` method to save and load the index, you can use the index in normal mode and low memory mode interchangeably. For example, you can build the index in normal mode, save it to disk with ``write`` method. After that, you can initialize the ``FlashEntropySearch`` object with ``path_data`` parameter which points to the index file, and set ``low_memory`` parameter to ``True``, then call the ``read`` method to load the index, and do the rest search as usual.


Run Flash entropy search with multiple cores
============================================

When you have many query spectra and your computer has multiple cores, you can use the multiple cores to speed up the search. You can use the built-in ``multiprocessing`` module to do this. The following code shows how to use the ``multiprocessing`` module to speed up the search.

To avoid the overhead of initializing the ``FlashEntropySearch`` object in each process, you can use the ``save_memory_for_multiprocessing`` method to save the memory for multiprocessing. This function will copy the index data to the shared memory to avoid the overhead of copying the index data to each process. After that, you can use the ``initializer`` and ``initargs`` parameters of the ``Pool`` class to initialize the ``FlashEntropySearch`` object in each process.

The following code shows an example how to use the ``multiprocessing`` module to calculate the maximum entropy similarity of 100 query spectra with 4 cores.

.. code-block:: python

    query_spectrum = {"precursor_mz": 150.0,
                      "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)}

    # Let's say you have 100 query spectra.
    query_spectra_list = [query_spectrum] * 100

    # And you want to use 4 cores to speed up the search.
    THREADS = 4

    entropy_search = FlashEntropySearch()
    entropy_search.build_index(spectral_library)

    # Call this function to save memory for multiprocessing.
    entropy_search.save_memory_for_multiprocessing()

    def func_max_similarity(precursor_mz, peaks):
        entropy_search = func_max_similarity.entropy_search
        similarity = entropy_search.search(precursor_mz=precursor_mz, peaks=peaks, method="neutral_loss")
        return np.max(similarity["neutral_loss_search"])

    def init_worker(entropy_search, ):
        func_max_similarity.entropy_search = entropy_search
        return None

    pool = mp.Pool(THREADS, initializer=init_worker, initargs=(entropy_search, ))
    max_entropy_similarity = pool.starmap(func_max_similarity, [(spectrum["precursor_mz"], spectrum["peaks"]) for spectrum in query_spectra_list])

.. note:: 
    When using the multiple cores, you always need to keep in mind of the memory usage.
    
    Let's say if you want to search 1,000,000 spectra against a spectral library contains 1,000,000 spectra. The result similarity matrix will be **4*1,000,000*1,000,000 = 3.6 TB!** Therefore, returning the whole similairty matrix is not a good idea.
    
    Instead, you can do some process on the result similarity, and only return the processed result. This will save a lot of memory, and a lot of computation time on copying large memory. For example, if you only return the top 10 similarity for each query spectrum, the memory usage will be **4*1,000,000*10 = 38 MB**. This is will be much more efficient.


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
    # )
    # Use this:
    entropy_similarity = entropy_search.search(
        precursor_mz=150.0,
        peaks=[[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]],
        target='gpu'
    )

    # Then the reset of the code is the same as usual.

The return value of calculating with ``CPU`` and ``GPU`` is the same. You can use the same code to process the result.