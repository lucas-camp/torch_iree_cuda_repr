#!/bin/bash

echo "LINALG"
python3 compare.py output/torch-mlir/padding_1_1/expected_output.npy output/torch-mlir/padding_1_1/linalg/cpu_output.npy
python3 compare.py output/torch-mlir/padding_1_1/expected_output.npy output/torch-mlir/padding_1_1/linalg/cuda_output.npy
python3 compare.py output/torch-mlir/padding_2_2/expected_output.npy output/torch-mlir/padding_2_2/linalg/cpu_output.npy
python3 compare.py output/torch-mlir/padding_2_2/expected_output.npy output/torch-mlir/padding_2_2/linalg/cuda_output.npy

echo "TOSA"
python3 compare.py output/torch-mlir/padding_1_1/expected_output.npy output/torch-mlir/padding_1_1/tosa/cpu_output.npy
python3 compare.py output/torch-mlir/padding_1_1/expected_output.npy output/torch-mlir/padding_1_1/tosa/cuda_output.npy
python3 compare.py output/torch-mlir/padding_2_2/expected_output.npy output/torch-mlir/padding_2_2/tosa/cpu_output.npy
python3 compare.py output/torch-mlir/padding_2_2/expected_output.npy output/torch-mlir/padding_2_2/tosa/cuda_output.npy
