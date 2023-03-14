#!/usr/bin/env python3
import pickle
import numpy as np
import time
import pandas as pd

from mimas.helper.arguments import Arguments
from library.entropy_search import EntropySearchCore
from mimas.helper.multiplecore import convert_numpy_array_to_shared_memory, MPRunner


def main(para):
    ms2_tolerance_in_da = 0.02

    ########################################################################################
    # Step 1: Read the query spectra.
    spectral_query = pickle.load(open(para.file_query, "rb"))
    spectral_library = pickle.load(open(para.file_library, "rb"))

    ########################################################################################
    # Step 2: Build the index for fast methods.
    # For fast entropy search.
    entropy_search = EntropySearchCore()
    entropy_search.build_index(all_peaks_list=[x["peaks"] for x in spectral_library])

    all_result = []
    for entropy, spectral_list in spectral_query.items():
        for i, spec in enumerate(spectral_list):
            start_time = time.time()
            similarity = entropy_search.search(peaks=spec["peaks"], ms2_tolerance_in_da=ms2_tolerance_in_da, search_type=0, target="cpu")
            used_time = time.time() - start_time
            all_result.append({
                "entropy": entropy,
                "spec_id": i,
                "time": used_time,
                "library_size": len(spectral_library),
            })
    
    df=pd.DataFrame(all_result)
    df.to_csv(para.file_output, index=False)

    return 0


if __name__ == '__main__':
    args = Arguments()
    para = {
        "file_query": r"/p/FastEntropySearch/benchmark/data/public_repository/spectral_library_different_entropy/spectra-charge_N-number_100.pkl",
        "file_library": r"/p/FastEntropySearch/benchmark/data/public_repository/spectral_library/spectra-charge_N-number_1000000.pkl",
        "file_output": r"/p/FastEntropySearch/benchmark/test/test.csv",
    }
    args.add_argument_from_dictionary(para)
    para = args.parse_args(print_parameter=False)
    main(para)
