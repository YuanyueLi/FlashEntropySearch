#!/usr/bin/env python3
import pickle
import numpy as np
import time
import pandas as pd

from mimas.helper.arguments import Arguments
from library.entropy_search import EntropySearchCore
import multiprocessing as mp


def func_benchmark(peaks):
    entropy_search = func_benchmark.entropy_search
    ms2_tolerance_in_da = func_benchmark.ms2_tolerance_in_da
    similarity = entropy_search.search(peaks=peaks, ms2_tolerance_in_da=ms2_tolerance_in_da, search_type=0, target="cpu")
    return similarity[0]

def main(para):
    ms2_tolerance_in_da = 0.02

    ########################################################################################
    # Step 1: Read the query spectra.
    spectra_query = [x["peaks"] for x in pickle.loads(open(para.file_query, "rb").read())]

    ########################################################################################
    # Step 2: Build the index for fast methods.
    # For fast entropy search.
    entropy_search = EntropySearchCore()
    entropy_search.read_from_file(open(para.file_index, "rb"))
    
    def init_worker(entropy_search, ms2_tolerance_in_da):
        func_benchmark.entropy_search = entropy_search
        func_benchmark.ms2_tolerance_in_da = ms2_tolerance_in_da
        return None
    ########################################################################################
    if para["threads"] > 1:
        entropy_search.move_index_array_to_shared_memory()
        pool = mp.Pool(para["threads"], initializer=init_worker, initargs=(entropy_search, ms2_tolerance_in_da))
        start_time = time.time()
        pool.starmap(func_benchmark, [(peaks,) for peaks in spectra_query])
        used_time = time.time() - start_time
        print("Fast entropy search with multiprocessing used time: {}".format(used_time))
    else:
        # For fast entropy search.
        init_worker(entropy_search, ms2_tolerance_in_da)
        start_time = time.time()
        for spec in spectra_query:
            func_benchmark(spec)
        used_time = time.time() - start_time
        print("Fast entropy search used time: {}".format(used_time))
    return 0


if __name__ == '__main__':
    args = Arguments()
    para = {
        "threads": 8,
        "file_query": r"/p/FastEntropySearch/benchmark/data/mona/spectral_library/spectra-charge_N-number_100000.pkl",
        "file_index": r"/p/FastEntropySearch/benchmark/data/mona/library_index/index-charge_N-number_1000000.bin",
    }
    args.add_argument_from_dictionary(para)
    para = args.parse_args(print_parameter=False)
    main(para)
