#!/bin/bash

PATH_WORK=$(dirname "$0")
PYTHON="/home/yli/Software/anaconda3/envs/py39/bin/python"
SCRIPT="${PATH_WORK}/31_benchmark_with_multiple_cores.py"

# Copy the input file to the cache directory
# THREADS=1
QUERY_NUMBER=100000
LIBRARY_NUMBER=1000000
# SOURCE="public_repository"

for SOURCE in "mona" "gnps" "public_repository"; do
    PATH_DATA="${PATH_WORK}/../data/${SOURCE}/"
    for CHARGE in "P" "N"; do
        FILE_OUTPUT="${PATH_DATA}/multiple_cores/benchmark-charge_${CHARGE}.txt"
        mkdir -p $(dirname "${FILE_OUTPUT}")
        for THREADS in 1 2 4 8; do
            echo "Benchmarking with ${THREADS} threads, charge ${CHARGE}, library number ${LIBRARY_NUMBER} and query number ${QUERY_NUMBER}..."
            /usr/bin/time -v ${PYTHON} "${SCRIPT}" \
                -file_index "${PATH_DATA}/library_index/index-charge_${CHARGE}-number_${LIBRARY_NUMBER}.bin" \
                -file_query "${PATH_DATA}/spectral_library/spectra-charge_${CHARGE}-number_${QUERY_NUMBER}.pkl" \
                -threads ${THREADS}
        done 2>&1 | tee "${FILE_OUTPUT}"
    done
done
echo "Finished!"
