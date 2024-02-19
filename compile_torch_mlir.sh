#!/bin/bash

python3 generate_mlir_module_torch_mlir.py --pad_x=1 --pad_y=1
python3 generate_mlir_module_torch_mlir.py --pad_x=2 --pad_y=2

iree-compile \
  --iree-hal-target-backends=llvm-cpu \
  output/torch-mlir/padding_1_1/linalg/module.mlir \
  -o output/torch-mlir/padding_1_1/linalg/cpu.vmfb

iree-compile \
  --iree-hal-target-backends=cuda \
  --iree-hal-cuda-llvm-target-arch=sm_70 \
  output/torch-mlir/padding_1_1/linalg/module.mlir \
  -o output/torch-mlir/padding_1_1/linalg/cuda.vmfb

iree-compile \
  --iree-hal-target-backends=llvm-cpu \
  output/torch-mlir/padding_2_2/linalg/module.mlir \
  -o output/torch-mlir/padding_2_2/linalg/cpu.vmfb

iree-compile \
  --iree-hal-target-backends=cuda \
  --iree-hal-cuda-llvm-target-arch=sm_70 \
  output/torch-mlir/padding_2_2/linalg/module.mlir \
  -o output/torch-mlir/padding_2_2/linalg/cuda.vmfb

iree-compile \
  --iree-hal-target-backends=llvm-cpu \
  output/torch-mlir/padding_1_1/tosa/module.mlir \
  -o output/torch-mlir/padding_1_1/tosa/cpu.vmfb

iree-compile \
  --iree-hal-target-backends=cuda \
  --iree-hal-cuda-llvm-target-arch=sm_70 \
  output/torch-mlir/padding_1_1/tosa/module.mlir \
  -o output/torch-mlir/padding_1_1/tosa/cuda.vmfb

iree-compile \
  --iree-hal-target-backends=llvm-cpu \
  output/torch-mlir/padding_2_2/tosa/module.mlir \
  -o output/torch-mlir/padding_2_2/tosa/cpu.vmfb

iree-compile \
  --iree-hal-target-backends=cuda \
  --iree-hal-cuda-llvm-target-arch=sm_70 \
  output/torch-mlir/padding_2_2/tosa/module.mlir \
  -o output/torch-mlir/padding_2_2/tosa/cuda.vmfb