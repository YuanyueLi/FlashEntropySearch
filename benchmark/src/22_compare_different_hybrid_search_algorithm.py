#!/usr/bin/env python3
import pickle
import numpy as np
import time
import pandas as pd

from mimas.spectra.spectral_entropy_published.spectral_entropy import calculate_entropy_similarity
from mimas.helper.arguments import Arguments

from library.hybrid_search import EntropyHybridSearchCore
from library import blink
import matchms as mms
from matchms.similarity import CosineGreedy


def main(para):
    ms2_tolerance_in_da = 0.02
    blink_bin_width = 0.001
    intensity_power = 0.5

    ########################################################################################
    # Step 1: Read the query spectra and library spectra.
    spectra_query_raw = pickle.loads(open(para.file_query, "rb").read())
    spectra_query = [x["peaks"] for x in spectra_query_raw]

    spectra_library_raw = pickle.loads(open(para.file_library, "rb").read())
    spectra_library = [x["peaks"] for x in spectra_library_raw]
    spectral_library_length = len(spectra_library)

    ########################################################################################
    # Step 2: Build the index for fast methods.
    # For fast entropy search.
    entropy_search = EntropyHybridSearchCore()
    entropy_search.build_index(all_precursor_mz_list=[x["precursor_mz"] for x in spectra_library_raw],
                               all_peaks_list=spectra_library)

    # For matchms.
    spectra_library_for_matchms = [
        mms.Spectrum(mz=spec['peaks'][:, 0].astype("float"), intensities=spec['peaks'][:, 1].astype("float"), metadata={"precursor_mz": spec['precursor_mz']})
        for spec in spectra_library_raw]
    spectra_query_for_matchms = [
        mms.Spectrum(mz=spec['peaks'][:, 0].astype("float"), intensities=spec['peaks'][:, 1].astype("float"), metadata={"precursor_mz": spec['precursor_mz']})
        for spec in spectra_query_raw]

    # ########################################################################################
    # # Step 3: Benchmark the similarity calculation.
    result = []
    for i, spec_query in enumerate(spectra_query):
        benchmark = {"id": i}
        # For fast entropy search.
        start_time = time.time()
        similarity_entropy_search = entropy_search.search(
            precursor_mz=spectra_query_raw[i]["precursor_mz"],
            peaks=spec_query, ms2_tolerance_in_da=ms2_tolerance_in_da, search_type=0, target="cpu")
        used_time = time.time() - start_time
        benchmark["fast_entropy_search"] = used_time

        # For matchms.
        cos = CosineGreedy(tolerance=ms2_tolerance_in_da, intensity_power=0.5)
        start_time = time.time()
        similarity_matchms = cos.matrix(queries=[spectra_query_for_matchms[i]], references=spectra_library_for_matchms).flatten()
        used_time = time.time() - start_time
        benchmark["matchms"] = used_time

        result.append(benchmark)

    # Step 5: Save the result.
    df = pd.DataFrame(result)
    df["query_number"] = len(spectra_query)
    df["library_number"] = len(spectra_library)
    df.to_csv(para["file_output"], index=False)
    return 0


if __name__ == '__main__':
    args = Arguments()
    para = {
        "file_query": r"/p/FastEntropySearch/benchmark/data/mona/spectral_library/spectra-charge_N-number_100.pkl",
        "file_library": r"/p/FastEntropySearch/benchmark/data/mona/spectral_library/spectra-charge_N-number_100.pkl",
        "file_output": r"/p/FastEntropySearch/benchmark/test/test.csv",
    }
    args.add_argument_from_dictionary(para)
    para = args.parse_args(print_parameter=True)
    main(para)
