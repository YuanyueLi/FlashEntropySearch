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
from matchms.similarity import NeutralLossesCosine


def main(para):
    ms2_tolerance_in_da = 0.02
    blink_bin_width = 0.001
    intensity_power = 0.5

    ########################################################################################
    # Step 1: Read the query spectra and library spectra.
    spectra_query_raw = pickle.loads(open(para.file_query, "rb").read())

    spectra_library_raw = pickle.loads(open(para.file_library, "rb").read())
    spectra_library_raw_length = len(spectra_library_raw)

    # Calculate the neutral loss library.
    spectra_library_nl = []
    spectra_library_raw_new = []
    for spec in spectra_library_raw:
        # Remove error spectra, which will cause BLINK runs too slow.
        if spec["precursor_mz"] > 3000:
            continue
        spectra_library_raw_new.append(spec)
        peaks = np.copy(spec["peaks"])
        peaks[:, 0] = spec["precursor_mz"]-peaks[:, 0]
        # Sort the peaks by mz.
        peaks = peaks[np.argsort(peaks[:, 0])]
        spectra_library_nl.append({"peaks": peaks, "precursor_mz": spec["precursor_mz"]})
    spectra_library_raw = spectra_library_raw_new
    spectral_library_length = len(spectra_library_raw)

    ########################################################################################
    # Step 2: Build the index for fast methods.
    # For fast entropy search.
    entropy_search = EntropySearchCore()
    entropy_search.build_index([x["peaks"] for x in spectra_library_nl])

    # For blink.
    spectra_library_peaks_for_blink = [x["peaks"].T for x in spectra_library_nl]
    spectra_library_mz_for_blink = [x["precursor_mz"] for x in spectra_library_nl]
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
    for i, spec_query in enumerate(spectra_query_raw):
        peaks_nl = np.copy(spec_query["peaks"])
        peaks_nl[:, 0] = spec_query["precursor_mz"]-peaks_nl[:, 0]
        # Sort the peaks by mz.
        peaks_nl = peaks_nl[np.argsort(peaks_nl[:, 0])]

        benchmark = {"id": i}
        # For fast entropy search.
        start_time = time.time()
        similarity_entropy_search = entropy_search.search(peaks=peaks_nl, ms2_tolerance_in_da=ms2_tolerance_in_da, search_type=0, target="cpu")
        used_time = time.time() - start_time
        benchmark["fast_entropy_search"] = used_time

        # For native entropy search.
        similarity_traditional = np.zeros((spectral_library_length), dtype=np.float32)
        start_time = time.time()
        for j in range(spectral_library_length):
            similarity_traditional[j] = calculate_entropy_similarity(peaks_nl, spectra_library_nl[j]["peaks"], ms2_da=0.02, need_clean_spectra=False)
        used_time = time.time() - start_time
        benchmark["native_entropy_search"] = used_time

        max_difference = np.max(np.abs(similarity_entropy_search - similarity_traditional))
        benchmark["max_difference_for_fast_entropy_search"] = max_difference

        # For blink.
        blink_search_S1 = blink.discretize_spectra(
            [peaks_nl.T], [spec_query["precursor_mz"]], intensity_power=intensity_power, calc_network_score=False, bin_width=blink_bin_width)
        start_time = time.time()
        S12 = blink.score_sparse_spectra(blink_search_S1, blink_search_S2, tolerance=ms2_tolerance_in_da, calc_network_score=False)
        used_time = time.time() - start_time
        similarity_blink = S12['mzi'].toarray().flatten()
        benchmark["blink"] = used_time

        # For matchms.
        cos = NeutralLossesCosine(tolerance=ms2_tolerance_in_da, intensity_power=intensity_power)
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
    df["query_number"] = len(spectra_query_raw)
    df["library_number"] = spectra_library_raw_length
    df.to_csv(para["file_output"], index=False)
    return 0


if __name__ == '__main__':
    args = Arguments()
    para = {
        "file_query": r"/p/FastEntropySearch/benchmark/data/mona/spectral_library/spectra-charge_N-number_100.pkl",
        "file_library": r"/p/FastEntropySearch/benchmark/data/public_repository/spectral_library/spectra-charge_N-number_100.pkl",
        "file_output": r"/p/FastEntropySearch/benchmark/test/test.csv",
    }
    args.add_argument_from_dictionary(para)
    para = args.parse_args(print_parameter=True)
    main(para)
