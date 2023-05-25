#!/usr/bin/env python
import pickle
import pandas as pd
import time

from mimas.helper.arguments import Arguments
from library.entropy_search_universe import EntropySearchUniverseHybrid


def main(para):
    ms2_tolerance_in_da = 0.02
    entropy_search_universe = EntropySearchUniverseHybrid(para.path_index, mode="read")
    all_spectra = pickle.load(open(para.file_spectra, "rb"))

    target = para["target"]
    all_result = []
    for idx, spec in enumerate(all_spectra):
        # Identity search
        time_start = time.time()
        result = entropy_search_universe.search_hybrid(
            charge=spec["charge"], precursor_mz=spec["precursor_mz"],
            peaks=spec["peaks"], ms2_tolerance_in_da=ms2_tolerance_in_da, target=target)
        time_hybrid = time.time() - time_start

        all_result.append({
            "idx": idx,
            "target": target,
            "time_hybrid": time_hybrid,
            "number_of_spectra": len(result)
        })

    df = pd.DataFrame(all_result)
    df.to_csv(para["file_output"], index=False)
    return 0


if __name__ == '__main__':
    args = Arguments()
    para = {
        "target": "gpu",
        "file_spectra": r"/p/FastEntropySearch/benchmark/data/public_repository/spectral_library/spectra-charge_N-number_100.pkl",
        "path_index": r"/p/Atlas/scripts/application/20_search_whole_universe/test/hybrid_index",
        "file_output": r"/p/FastEntropySearch/benchmark/test/test.csv"
    }

    args.add_argument_from_dictionary(para)
    para = args.parse_args(print_parameter=True)
    main(para)
    print("Done")
