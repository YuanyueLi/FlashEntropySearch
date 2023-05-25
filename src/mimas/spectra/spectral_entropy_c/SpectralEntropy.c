#include "CleanSpectrum.h"
#include "SpectralEntropy.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float_spec inline minus_xlog2x(float_spec x) {
	return -x * log2f(x);
}


// Calculate spectral entropy of a spectrum. The spectrum intensity need to be prenormalized.
float_spec calculate_spectral_entropy(const float_spec* spectrum, int spectrum_length) {
	float_spec entropy = 0;
	float_spec(*spectrum_array_2d)[2] = (float_spec(*)[2])spectrum;
	float intensity;
	int i;
	for (i = 0; i < spectrum_length; i++) {
		intensity = spectrum_array_2d[i][1];
		entropy -= intensity * logf(intensity);
	}
	return entropy;
}

// Apply weight to a spectrum by spectral entropy.
void apply_weight_to_intensity(float_spec* spectrum, int spectrum_length) {
	float_spec(*spectrum_array_2d)[2] = (float_spec(*)[2])spectrum;
	float_spec entropy = calculate_spectral_entropy(spectrum, spectrum_length);
	if (entropy < 3) {
		const float_spec weight = 0.25 + 0.25 * entropy;
		float_spec intensity_sum = 0;
		int i;
		for (i = 0; i < spectrum_length; i++) {
			spectrum_array_2d[i][1] = powf(spectrum_array_2d[i][1], weight);
			intensity_sum += spectrum_array_2d[i][1];
		}
		if (intensity_sum > 0) {
			for (i = 0; i < spectrum_length; i++) {
				spectrum_array_2d[i][1] /= intensity_sum;
			}
		}
	}
}

// Calculate unweighted entropy similarity
float unweighted_entropy_similarity(
	float_spec* spectrum_a, int spectrum_a_len,
	float_spec* spectrum_b, int spectrum_b_len,
	float ms2_da, bool spectra_is_preclean
) {
	if (!spectra_is_preclean) {
		clean_spectrum(spectrum_a, &spectrum_a_len, 0, -1, 0.01, -1, true, ms2_da);
		clean_spectrum(spectrum_b, &spectrum_b_len, 0, -1, 0.01, -1, true, ms2_da);
	}

	float_spec(*spec_a_2d)[2] = (float_spec(*)[2])spectrum_a;
	float_spec(*spec_b_2d)[2] = (float_spec(*)[2])spectrum_b;
	if (__DEBUG__ENTROPY_SIMILARTY__) print_spectrum("spec_query:\n", spec_a_2d, spectrum_a_len);
	if (__DEBUG__ENTROPY_SIMILARTY__) print_spectrum("spec_reference:\n", spec_b_2d, spectrum_b_len);
	if (spectrum_a_len == 0 || spectrum_b_len == 0) {
		return 0.0;
	}

	int a = 0, b = 0;
	float_spec peak_b_int = 0;
	float_spec entropy_merged = 0;
	float_spec entropy_a = 0;
	float_spec entropy_b = 0;
	float_spec peak_merged = 0;

	while (a < spectrum_a_len && b < spectrum_b_len) {
		float mass_delta_da = spec_a_2d[a][0] - spec_b_2d[b][0];
		if (mass_delta_da < -ms2_da) {
			// Peak only existed in spec a.
			entropy_a += minus_xlog2x(spec_a_2d[a][1]);
			if (peak_b_int > 0) {
				entropy_b += minus_xlog2x(peak_b_int);
			}
			peak_merged = (spec_a_2d[a][1] + peak_b_int) / 2;
			entropy_merged += minus_xlog2x(peak_merged);
			peak_b_int = 0;
			a++;
		}
		else if (mass_delta_da > ms2_da) {
			// Peak only existed in spec b.
			entropy_b += minus_xlog2x(spec_b_2d[b][1]);
			peak_merged = (spec_b_2d[b][1]) / 2;
			entropy_merged += minus_xlog2x(peak_merged);
			b++;
		}
		else {
			// Peak existed in both spec a and spec b.
			peak_b_int += spec_b_2d[b][1];
			b++;
		}

	}
	if (peak_b_int > 0) {
		entropy_a += minus_xlog2x(spec_a_2d[a][1]);
		entropy_b += minus_xlog2x(peak_b_int);
		peak_merged = (spec_a_2d[a][1] + peak_b_int) / 2;
		entropy_merged += minus_xlog2x(peak_merged);

		peak_b_int = 0;
		a++;
	}

	// Fill the rest into merged spec
	int i;
	for (i = b; i < spectrum_b_len; i++) {
		entropy_b += minus_xlog2x(spec_b_2d[i][1]);
		peak_merged = spec_b_2d[i][1] / 2;
		entropy_merged += minus_xlog2x(peak_merged);
	}
	for (i = a; i < spectrum_a_len; i++) {
		entropy_a += minus_xlog2x(spec_a_2d[i][1]);
		peak_merged = spec_a_2d[i][1] / 2;
		entropy_merged += minus_xlog2x(peak_merged);
	}
	return 1 - entropy_merged + (entropy_a + entropy_b) / 2;
}

// Calculate entropy similarity
float entropy_similarity(
	float_spec* spectrum_a, int spectrum_a_len,
	float_spec* spectrum_b, int spectrum_b_len,
	float ms2_da, bool spectra_is_preclean
) {
	if (!spectra_is_preclean) {
		clean_spectrum(spectrum_a, &spectrum_a_len, 0, -1, 0.01, -1, true, ms2_da);
		clean_spectrum(spectrum_b, &spectrum_b_len, 0, -1, 0.01, -1, true, ms2_da);
	}
	if (__DEBUG__ENTROPY_SIMILARTY__) print_spectrum("spec_query:\n", (float_spec(*)[2])spectrum_a, spectrum_a_len);
	if (__DEBUG__ENTROPY_SIMILARTY__) print_spectrum("spec_reference:\n", (float_spec(*)[2])spectrum_b, spectrum_b_len);
	if (spectrum_a_len == 0 || spectrum_b_len == 0) {
		return 0.0;
	}
	apply_weight_to_intensity(spectrum_a, spectrum_a_len);
	apply_weight_to_intensity(spectrum_b, spectrum_b_len);
	return unweighted_entropy_similarity(spectrum_a, spectrum_a_len, spectrum_b, spectrum_b_len, ms2_da, true);
}
