#!/bin/bash

PATH_WORK=$(dirname "$0")
PYTHON="/home/yli/Software/anaconda3/envs/py39/bin/python"
SCRIPT="${PATH_WORK}/40_benchmark_search_whole_repository-identity.py"

# Copy the input file to the cache directory
QUERY_NUMBER=100
SOURCE="public_repository"
PATH_DATA="${PATH_WORK}/../data/${SOURCE}/"
PATH_INDEX="/p/Atlas/resource/index/identity_index/"

CHARGE="P"
for TARGET in "gpu" "cpu"; do
    systemd-run --scope -p MemoryMax=4G ${PYTHON} "${SCRIPT}" \
        -target "${TARGET}" \
        -file_spectra "${PATH_DATA}/spectral_library/spectra-charge_${CHARGE}-number_${QUERY_NUMBER}.pkl" \
        -path_index "${PATH_INDEX}" \
        -file_output "${PATH_DATA}/whole_repository/benchmark-identity-charge_${CHARGE}-number_${QUERY_NUMBER}-target_${TARGET}.csv"
done

CHARGE="N"
for TARGET in "gpu" "cpu"; do
    systemd-run --scope -p MemoryMax=4G ${PYTHON} "${SCRIPT}" \
        -target "${TARGET}" \
        -file_spectra "${PATH_DATA}/spectral_library/spectra-charge_${CHARGE}-number_${QUERY_NUMBER}.pkl" \
        -path_index "${PATH_INDEX}" \
        -file_output "${PATH_DATA}/whole_repository/benchmark-identity-charge_${CHARGE}-number_${QUERY_NUMBER}-target_${TARGET}.csv"
done

echo "Finished!"
