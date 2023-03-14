#include "CleanSpectrum.h"
#include <stdio.h>
#include <stdlib.h>


// Print spectrum 2d array
void print_spectrum(char* info, float_spec(*spectrum_2d)[2], int spectrum_len)
{
	printf("%s", info);
	int i;
	for (i = 0; i < spectrum_len; i++) {
		printf("%d\t%f\t%f\n", i, spectrum_2d[i][0], spectrum_2d[i][1]);
	}
};

void swap(float_spec* a, float_spec* b) {
	float_spec c = *a;
	*a = *b;
	*b = c;
};
void inline swap_int(int* a, int* b) {
	int c = *a;
	*a = *b;
	*b = c;
}

void inline sort_spectrum_by_mz(float_spec(*spectrum_2d)[2], int spectrum_len) {
	int i,j;
	for (i = 0; i < spectrum_len; i++) {
		for (j = i + 1; j < spectrum_len; j++) {
			if (spectrum_2d[i][0] > spectrum_2d[j][0]) {
				swap(&(spectrum_2d[i][0]), &(spectrum_2d[j][0]));
				swap(&(spectrum_2d[i][1]), &(spectrum_2d[j][1]));
			}
		}
	}
}

void sort_spectrum_by_mz_and_zero_intensity(float_spec(*spectrum_2d)[2], int* spectrum_len) {
	for (int i = 0; i < *spectrum_len; i++) {
		for (int j = i + 1; j < *spectrum_len; j++) {
			if ((spectrum_2d[j][1] > 0) &&
				(spectrum_2d[i][1] <= 0 || spectrum_2d[i][0] > spectrum_2d[j][0])) {
				swap(&(spectrum_2d[i][0]), &(spectrum_2d[j][0]));
				swap(&(spectrum_2d[i][1]), &(spectrum_2d[j][1]));
			}
		}
	}
	while (*spectrum_len >= 1 && spectrum_2d[*spectrum_len - 1][1] <= 0) {
		(*spectrum_len)--;
	}
}

void inline calculate_spectrum_argsort(float_spec(*spectrum_2d)[2], int spectrum_len, int* spectrum_argsort) {
	for (int i = 0; i < spectrum_len; i++) {
		spectrum_argsort[i] = i;
	}
	for (int i = 0; i < spectrum_len - 1; i++) {
		for (int j = 0; j < spectrum_len - 1 - i; j++) {
			int* a = &(spectrum_argsort[j]), * b = &(spectrum_argsort[j + 1]);
			if (spectrum_2d[*a][1] < spectrum_2d[*b][1]) {
				swap_int(a, b);
			}
		}
	}
}

bool inline need_centroid(float_spec(*spectrum_2d)[2], int spectrum_len, float ms2_da) {
	for (int i = 0; i < spectrum_len - 1; i++) {
		if (spectrum_2d[i + 1][0] - spectrum_2d[i][0] < ms2_da) {
			return true;
		}
	}
	return false;
}

// Centroid the spectrum, the content in the spectrum will be modified.
int inline centroid_spectrum(float_spec(*spectrum_2d)[2], int* spectrum_length, float_spec ms2_da, int* spectrum_argsort) {
	// Calculate the argsort of the spectrum by intensity.
	calculate_spectrum_argsort(spectrum_2d, *spectrum_length, spectrum_argsort);

	// Centroid the spectrum.
	for (int i = 0; i < *spectrum_length; i++) {
		int idx = spectrum_argsort[i];
		if (spectrum_2d[idx][1] > 0) {
			// Find left board for current peak
			int idx_left = idx - 1;
			while (idx_left >= 0 && spectrum_2d[idx][0] - spectrum_2d[idx_left][0] <= ms2_da) {
				idx_left--;
			}

			// Find right board for current peak
			int idx_right = idx + 1;
			while (idx_right < *spectrum_length && spectrum_2d[idx_right][0] - spectrum_2d[idx][0] <= ms2_da) {
				idx_right++;
			}

			// Merge the peaks in the board
			float_spec intensity_sum = 0;
			float_spec intensity_weighted_sum = 0;
			if (__DEBUG__CLEAN_SPECTRUM__) printf("%d\t%d\t%d\n", idx, idx_left+1, idx_right-1);
			if (__DEBUG__CLEAN_SPECTRUM__) printf("%f\t%f\t%f\t%f\n", spectrum_2d[idx][0], spectrum_2d[idx][1], spectrum_2d[idx_left+1][0], spectrum_2d[idx_right-1][0]);
			for (int i = idx_left + 1; i < idx_right; i++) {
				intensity_sum += spectrum_2d[i][1];
				intensity_weighted_sum += spectrum_2d[i][1] * spectrum_2d[i][0];
				spectrum_2d[i][1] = 0;
			}

			// Write the new peak into the output spectrum
			spectrum_2d[idx][0] = intensity_weighted_sum / intensity_sum;
			spectrum_2d[idx][1] = intensity_sum;
		}
	}
	sort_spectrum_by_mz_and_zero_intensity(spectrum_2d, spectrum_length);
	return 0;
}

// Clean the spectrum.
// The spectrum is a 2D array. spectrum[x][0] is the m/z, spectrum[x][1] is the intensity.
// The spectrum will be rewritten.
int clean_spectrum(float_spec* spectrum, int* spectrum_length,
	float min_mz, float max_mz,
	float noise_threshold,
	int max_peak_num,
	bool normalize_intensity,
	float ms2_da) {
	float_spec(*spectrum_2d)[2] = (float_spec(*)[2]) & spectrum[0];
	int* spectrum_argsort = (int*)malloc(*spectrum_length * sizeof(int));

	if (__DEBUG__CLEAN_SPECTRUM__) print_spectrum("Input:\n", spectrum_2d, *spectrum_length);
	// 1. Remove the peaks by m/z.
	for (int i = 0; i < *spectrum_length; i++) {
		if (spectrum_2d[i][0] <= min_mz ||
			(max_mz > 0 && spectrum_2d[i][0] >= max_mz)) {
			spectrum_2d[i][1] = 0;
		}
	}
	sort_spectrum_by_mz_and_zero_intensity(spectrum_2d, spectrum_length);

	if (__DEBUG__CLEAN_SPECTRUM__) print_spectrum("Remove the peaks by m/z:\n", spectrum_2d, *spectrum_length);

	// 2. Centroid the spectrum.
	while (need_centroid(spectrum_2d, *spectrum_length, ms2_da)) {
		centroid_spectrum(spectrum_2d, spectrum_length, ms2_da, spectrum_argsort);
		if (__DEBUG__CLEAN_SPECTRUM__) print_spectrum("Centroid the spectrum:\n", spectrum_2d, *spectrum_length);
	}
	// 3. Remove the peaks with intensity less than the noise_threshold * maximum(intensity).
	if (noise_threshold > 0) {
		float_spec max_intensity = 0;
		for (int i = 0; i < *spectrum_length; i++) {
			if (spectrum_2d[i][1] > max_intensity) {
				max_intensity = spectrum_2d[i][1];
			}
		}
		float_spec noise_threshold_intensity = noise_threshold * max_intensity;
		if (__DEBUG__CLEAN_SPECTRUM__) printf("Remove the peaks with intensity less than %f * %f = %f:\n", noise_threshold, max_intensity, noise_threshold_intensity);
		for (int i = 0; i < *spectrum_length; i++) {
			if (spectrum_2d[i][1] < noise_threshold_intensity) {
				spectrum_2d[i][1] = 0;
			}
		}
		if (__DEBUG__CLEAN_SPECTRUM__) print_spectrum("Remove the noise:\n", spectrum_2d, *spectrum_length);
	}

	// 4. Select top K peaks.
	if (max_peak_num > 0 && max_peak_num < *spectrum_length) {
		calculate_spectrum_argsort(spectrum_2d, *spectrum_length, spectrum_argsort);
		for (int i = max_peak_num; i < *spectrum_length; i++) {
			spectrum_2d[spectrum_argsort[i]][1] = 0;
		}
		if (__DEBUG__CLEAN_SPECTRUM__) print_spectrum("Select top K peaks:\n", spectrum_2d, *spectrum_length);
	}
	sort_spectrum_by_mz_and_zero_intensity(spectrum_2d, spectrum_length);

	// 5. Normalize the intensity to sum to 1.
	if (normalize_intensity) {
		float_spec sum_intensity = 0;
		for (int i = 0; i < *spectrum_length; i++) {
			sum_intensity += spectrum_2d[i][1];
		}
		if (sum_intensity > 0) {
			for (int i = 0; i < *spectrum_length; i++) {
				spectrum_2d[i][1] /= sum_intensity;
			}
		}
		if (__DEBUG__CLEAN_SPECTRUM__) print_spectrum("Normalize the intensity:\n", spectrum_2d, *spectrum_length);
	}

	free(spectrum_argsort);
	return 0;
}
