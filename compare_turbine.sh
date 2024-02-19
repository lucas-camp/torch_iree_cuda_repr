#!/bin/bash

python3 compare.py output/turbine/padding_1_1/expected_output.npy output/turbine/padding_1_1/cpu_output.npy
python3 compare.py output/turbine/padding_1_1/expected_output.npy output/turbine/padding_1_1/cuda_output.npy
python3 compare.py output/turbine/padding_2_2/expected_output.npy output/turbine/padding_2_2/cpu_output.npy
python3 compare.py output/turbine/padding_2_2/expected_output.npy output/turbine/padding_2_2/cuda_output.npy
