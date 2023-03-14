import numpy as np
from flash_entropy.entropy_search import EntropySearch
from flash_entropy.functions import clean_spectrum

# This is your library spectra, here the "precursor_mz" and "peaks" are required. The "id" is optional.
spectral_library = [
    {
        "id": "Demo spectrum 1",
        "precursor_mz": 150.0,
        "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [103.0, 1.0]], dtype=np.float32)
    },
    {
        "id": "Demo spectrum 2",
        "precursor_mz": 250.0,
        "peaks": np.array([[200.0, 1.0], [201.0, 1.0], [202.0, 1.0]], dtype=np.float32)
    },
    {
        "id": "Demo spectrum 3",
        "precursor_mz": 200.0,
        "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)
    }
]
# This is your query spectrum, here the "peaks" is required, the "precursor_mz" is required for identity search.
query_spectrum = {
    "precursor_mz": 150.0,
    "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)
}

#################### Step 1: Clean the spectra. ####################
# Clean the library spectra.
for spectrum in spectral_library:
    spectrum["peaks"] = clean_spectrum(peaks=spectrum['peaks'], max_mz=spectrum['precursor_mz']-1.6, noise_threshold=0.01, ms2_da=0.05)

# Clean the query spectrum.
query_spectrum["peaks"] = clean_spectrum(peaks=query_spectrum['peaks'], max_mz=query_spectrum['precursor_mz']-1.6, noise_threshold=0.01, ms2_da=0.05)

#################### Step 2: Build the library index. ####################
flash_entropy = EntropySearch()
# Please note that the library spectra will be re-sorted by precursor_mz for fast identity search.
spectral_library = flash_entropy.build_index(spectral_library, sort_by_precursor_mz=True)
# This step is optional, just used to show the spectrum id in the search results.
flash_entropy.library_id = [spectrum['id'] for spectrum in spectral_library]

#################### Step 3: Perform the Flash entropy search. ####################
# Perform the identity search.
entropy_similarity = flash_entropy.search_identity(precursor_mz=query_spectrum['precursor_mz'],
                                                   peaks=query_spectrum['peaks'],
                                                   ms1_tolerance_in_da=0.01, ms2_tolerance_in_da=0.02)
# Output the best match.
best_match = np.argmax(entropy_similarity)
print(f"Best identity search match: {flash_entropy.library_id[best_match]}, entropy similarity: {entropy_similarity[best_match]:.4f}")

# Perform the open search.
entropy_similarity = flash_entropy.search_open(peaks=query_spectrum['peaks'], ms2_tolerance_in_da=0.02)

# Output the best match.
best_match = np.argmax(entropy_similarity)
print(f"Best open search match: {flash_entropy.library_id[best_match]}, entropy similarity: {entropy_similarity[best_match]:.4f}")