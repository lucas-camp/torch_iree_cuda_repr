#!/bin/bash

iree-run-module \
  --device=local-task \
  --module=output/torch-mlir/padding_1_1/linalg/cpu.vmfb \
  --function=forward \
  --input=@output/module_input.npy \
  --output=@output/torch-mlir/padding_1_1/linalg/cpu_output.npy

iree-run-module \
  --device=cuda \
  --module=output/torch-mlir/padding_1_1/linalg/cuda.vmfb \
  --function=forward \
  --input=@output/module_input.npy \
  --output=@output/torch-mlir/padding_1_1/linalg/cuda_output.npy

iree-run-module \
  --device=local-task \
  --module=output/torch-mlir/padding_2_2/linalg/cpu.vmfb \
  --function=forward \
  --input=@output/module_input.npy \
  --output=@output/torch-mlir/padding_2_2/linalg/cpu_output.npy

iree-run-module \
  --device=cuda \
  --module=output/torch-mlir/padding_2_2/linalg/cuda.vmfb \
  --function=forward \
  --input=@output/module_input.npy \
  --output=@output/torch-mlir/padding_2_2/linalg/cuda_output.npy

iree-run-module \
  --device=local-task \
  --module=output/torch-mlir/padding_1_1/tosa/cpu.vmfb \
  --function=forward \
  --input=@output/module_input.npy \
  --output=@output/torch-mlir/padding_1_1/tosa/cpu_output.npy

iree-run-module \
  --device=cuda \
  --module=output/torch-mlir/padding_1_1/tosa/cuda.vmfb \
  --function=forward \
  --input=@output/module_input.npy \
  --output=@output/torch-mlir/padding_1_1/tosa/cuda_output.npy

iree-run-module \
  --device=local-task \
  --module=output/torch-mlir/padding_2_2/tosa/cpu.vmfb \
  --function=forward \
  --input=@output/module_input.npy \
  --output=@output/torch-mlir/padding_2_2/tosa/cpu_output.npy

iree-run-module \
  --device=cuda \
  --module=output/torch-mlir/padding_2_2/tosa/cuda.vmfb \
  --function=forward \
  --input=@output/module_input.npy \
  --output=@output/torch-mlir/padding_2_2/tosa/cuda_output.npy
