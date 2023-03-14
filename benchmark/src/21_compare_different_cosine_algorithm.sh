#!/bin/bash

PATH_WORK=$(dirname "$0")
PYTHON="/home/yli/Software/anaconda3/envs/py39/bin/python"
SCRIPT="${PATH_WORK}/21_compare_different_cosine_algorithm.py"

# Copy the input file to the cache directory
QUERY_NUMBER=100

SOURCE="mona"

for SOURCE in "mona" "gnps" "public_repository"; do
    PATH_LIBRARY="${PATH_WORK}/../data/${SOURCE}/spectral_library/"
    PATH_RESULT="${PATH_WORK}/../data/${SOURCE}/benchmark_cosine_result/"
    for LIBRARY_NUMBER in 100 1000 10000 100000 1000000; do
        for CHARGE in "P" "N"; do
            ${PYTHON} "${SCRIPT}" \
                -file_query "${PATH_LIBRARY}/spectra-charge_${CHARGE}-number_${QUERY_NUMBER}.pkl" \
                -file_library "${PATH_LIBRARY}/spectra-charge_${CHARGE}-number_${LIBRARY_NUMBER}.pkl" \
                -file_output "${PATH_RESULT}/charge_${CHARGE}-${QUERY_NUMBER}_against_${LIBRARY_NUMBER}.csv"
        done
    done
done

echo "Finished!"
