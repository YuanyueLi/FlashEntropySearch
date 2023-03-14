#!/usr/bin/env python3
import pickle
import numpy as np
import time
import pandas as pd

from mimas.spectra.spectral_entropy_published.spectral_entropy import calculate_entropy_similarity
from mimas.helper.arguments import Arguments

from library.entropy_search import EntropySearchCore
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
    entropy_search = EntropySearchCore()
    entropy_search.build_index(spectra_library)

    # For blink.
    spectra_library_peaks_for_blink = [x["peaks"].T for x in spectra_library_raw]
    spectra_library_mz_for_blink = [x["precursor_mz"] for x in spectra_library_raw]
    blink_search_S2 = blink.discretize_spectra(
        spectra_library_peaks_for_blink, spectra_library_mz_for_blink, intensity_power=intensity_power, calc_network_score=False, bin_width=blink_bin_width)

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
        similarity_entropy_search = entropy_search.search(peaks=spec_query, ms2_tolerance_in_da=ms2_tolerance_in_da, search_type=0, target="cpu")
        used_time = time.time() - start_time
        benchmark["fast_entropy_search"] = used_time

        # For native entropy search.
        similarity_traditional = np.zeros((spectral_library_length), dtype=np.float32)
        start_time = time.time()
        for j in range(spectral_library_length):
            similarity_traditional[j] = calculate_entropy_similarity(spec_query, spectra_library[j], ms2_da=0.02, need_clean_spectra=False)
        used_time = time.time() - start_time
        benchmark["native_entropy_search"] = used_time

        max_difference = np.max(np.abs(similarity_entropy_search - similarity_traditional))
        benchmark["max_difference_for_fast_entropy_search"] = max_difference

        # For blink.
        blink_search_S1 = blink.discretize_spectra(
            [spec_query.T], [spectra_query_raw[i]["precursor_mz"]], intensity_power=intensity_power, calc_network_score=False, bin_width=blink_bin_width)
        start_time = time.time()
        S12 = blink.score_sparse_spectra(blink_search_S1, blink_search_S2, tolerance=ms2_tolerance_in_da, calc_network_score=False)
        used_time = time.time() - start_time
        similarity_blink = S12['mzi'].toarray().flatten()
        benchmark["blink"] = used_time

        # For matchms.
        cos = CosineGreedy(tolerance=ms2_tolerance_in_da, intensity_power=0.5)
        start_time = time.time()
        similarity_matchms = cos.matrix(queries=[spectra_query_for_matchms[i]], references=spectra_library_for_matchms).flatten()
        used_time = time.time() - start_time
        benchmark["matchms"] = used_time

        similarity_matchms = np.array([x[0] for x in similarity_matchms])
        max_difference = np.max(np.abs(similarity_blink - similarity_matchms))
        benchmark["max_difference_for_blink"] = max_difference

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
        "file_library": r"/p/FastEntropySearch/benchmark/data/mona/spectral_library/spectra-charge_N-number_1000.pkl",
        "file_output": r"/p/FastEntropySearch/benchmark/test/test.csv",
    }
    args.add_argument_from_dictionary(para)
    para = args.parse_args(print_parameter=True)
    main(para)
