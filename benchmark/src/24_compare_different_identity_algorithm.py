#!/usr/bin/env python3
import pickle
import numpy as np
import time
import pandas as pd

from mimas.spectra.spectral_entropy_published.spectral_entropy import calculate_entropy_similarity
from mimas.helper.arguments import Arguments

from library.entropy_search import EntropySearch
from library import blink
import matchms as mms
from matchms.similarity import CosineGreedy


def main(para):
    ms1_tolerance_in_da = 0.01
    ms2_tolerance_in_da = 0.02
    blink_bin_width = 0.001
    intensity_power = 0.5

    ########################################################################################
    # Step 1: Read the query spectra and library spectra.
    spectra_query_raw = pickle.loads(open(para.file_query, "rb").read())

    spectra_library_raw = pickle.loads(open(para.file_library, "rb").read())
    spectral_library_length = len(spectra_library_raw)

    ########################################################################################
    # Step 2: Build the index for fast methods.
    # For fast entropy search.
    entropy_search = EntropySearch()
    entropy_search.build_index(spectra_library_raw)

    spectra_library_mz = np.array([x["precursor_mz"] for x in spectra_library_raw])
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
    for i, spec_query in enumerate(spectra_query_raw):
        benchmark = {"id": i}
        # For fast entropy search.
        start_time = time.time()
        similarity_entropy_search = entropy_search.search_identity(
            precursor_mz=spec_query["precursor_mz"], peaks=spec_query["peaks"],
            ms1_tolerance_in_da=ms1_tolerance_in_da, ms2_tolerance_in_da=ms2_tolerance_in_da, target="cpu")
        used_time = time.time() - start_time
        benchmark["fast_entropy_search"] = used_time

        # For native entropy search.
        similarity_traditional = np.zeros((spectral_library_length), dtype=np.float32)
        start_time = time.time()
        selected_spec = np.where(np.abs(spectra_library_mz - spec_query["precursor_mz"]) <= ms1_tolerance_in_da)[0]
        for j in selected_spec:
            similarity_traditional[j] = calculate_entropy_similarity(
                spec_query["peaks"], spectra_library_raw[j]["peaks"], ms2_da=0.02, need_clean_spectra=False)
        used_time = time.time() - start_time
        benchmark["native_entropy_search"] = used_time

        # For matchms.
        cos = CosineGreedy(tolerance=ms2_tolerance_in_da, intensity_power=0.5)
        start_time = time.time()
        # The spectral library need to be generated in this way, or the matchms will be very slow.
        selected_spec = np.where(np.abs(spectra_library_mz - spec_query["precursor_mz"]) <= ms1_tolerance_in_da)[0]
        searched_library = [spectra_library_for_matchms[j] for j in selected_spec]
        similarity_matchms = cos.matrix(queries=[spectra_query_for_matchms[i]], references=searched_library).flatten()
        used_time = time.time() - start_time
        benchmark["matchms"] = used_time
        benchmark["matchms_library_number"] = len(searched_library)

        result.append(benchmark)

    # Step 5: Save the result.
    df = pd.DataFrame(result)
    df["query_number"] = len(spectra_query_raw)
    df["library_number"] = len(spectra_library_raw)
    df.to_csv(para["file_output"], index=False)
    return 0


if __name__ == '__main__':
    args = Arguments()
    para = {
        "file_query": r"/p/FastEntropySearch/benchmark/data/mona/spectral_library/spectra-charge_N-number_100.pkl",
        "file_library": r"/p/FastEntropySearch/benchmark/data/mona/spectral_library/spectra-charge_N-number_100000.pkl",
        "file_output": r"/p/FastEntropySearch/benchmark/test/test.csv",
    }
    args.add_argument_from_dictionary(para)
    para = args.parse_args(print_parameter=True)
    main(para)
