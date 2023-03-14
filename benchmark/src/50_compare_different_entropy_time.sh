#!/bin/bash

PATH_WORK=$(dirname "$0")
PYTHON="/home/yli/Software/anaconda3/envs/py39/bin/python"
SCRIPT="${PATH_WORK}/50_compare_different_entropy_time.py"

# Copy the input file to the cache directory
QUERY_NUMBER=100
LIBRARY_NUMBER=1000000

# SOURCE="mona"
PATH_QUERY="${PATH_WORK}/../data/public_repository/spectral_library_different_entropy/"
for SOURCE in "mona" "gnps" "public_repository"; do
    PATH_LIBRARY="${PATH_WORK}/../data/${SOURCE}/spectral_library/"
    PATH_RESULT="${PATH_WORK}/../data/${SOURCE}/benchmark_different_entropy/"
    for CHARGE in "P" "N"; do
        ${PYTHON} "${SCRIPT}" \
            -file_query "${PATH_QUERY}/spectra-charge_${CHARGE}-number_${QUERY_NUMBER}.pkl" \
            -file_library "${PATH_LIBRARY}/spectra-charge_${CHARGE}-number_${LIBRARY_NUMBER}.pkl" \
            -file_output "${PATH_RESULT}/charge_${CHARGE}-${QUERY_NUMBER}_against_${LIBRARY_NUMBER}.csv"
    done
done

echo "Finished!"
