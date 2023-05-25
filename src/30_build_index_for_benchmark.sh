#!/bin/bash

PATH_WORK=$(dirname "$0")
PYTHON="/home/yli/Software/anaconda3/envs/py39/bin/python"
SCRIPT="${PATH_WORK}/30_build_index_for_benchmark.py"

# Copy the input file to the cache directory
LIBRARY_NUMBER=1000000
for SOURCE in "mona" "gnps" "public_repository"; do
    PATH_DATA="${PATH_WORK}/../data/${SOURCE}/"
    for CHARGE in "P" "N"; do
        ${PYTHON} "${SCRIPT}" \
            -file_library "${PATH_DATA}/spectral_library/spectra-charge_${CHARGE}-number_${LIBRARY_NUMBER}.pkl" \
            -file_index "${PATH_DATA}/library_index/index-charge_${CHARGE}-number_${LIBRARY_NUMBER}.bin"
    done
done
echo "Finished!"
