#!/usr/bin/env python3
import pickle
import numpy as np
import time
import pandas as pd

from mimas.helper.arguments import Arguments
from library.entropy_search import EntropySearchCore


def main(para):
    ms2_tolerance_in_da = 0.02

    ########################################################################################
    # Step 1: Read the query spectra and library spectra.
    spectra_library_raw = pickle.loads(open(para.file_library, "rb").read())
    spectra_library = [x["peaks"] for x in spectra_library_raw]

    ########################################################################################
    # Step 2: Build the index for fast methods.
    # For fast entropy search.
    entropy_search = EntropySearchCore()
    entropy_search.build_index(spectra_library)
    entropy_search.write_to_file(open(para.file_index, "wb"))
    return 0


if __name__ == '__main__':
    args = Arguments()
    para = {
        "file_library": r"/p/Atlas/scripts/fast_entropy_search/10_benchmark_on_different_number_spectra/data/spectra_library/random_spectra-charge_-1-number_1000000.pkl",
        "file_index": r"/p/Atlas/scripts/fast_entropy_search/10_benchmark_on_different_number_spectra/test/benchmark_result/library_index.bin",
    }
    args.add_argument_from_dictionary(para)
    para = args.parse_args(print_parameter=True)
    main(para)
