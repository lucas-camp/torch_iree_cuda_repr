#!/bin/bash

python3 generate_mlir_module.py --pad_x=1 --pad_y=1
python3 generate_mlir_module.py --pad_x=2 --pad_y=2

iree-compile \
  --iree-hal-target-backends=llvm-cpu \
  output/padding_1_1/module.mlir \
  -o output/padding_1_1/cpu.vmfb

iree-compile \
  --iree-hal-target-backends=cuda \
  --iree-hal-cuda-llvm-target-arch=sm_70 \
  output/padding_1_1/module.mlir \
  -o output/padding_1_1/cuda.vmfb

iree-compile \
  --iree-hal-target-backends=llvm-cpu \
  output/padding_2_2/module.mlir \
  -o output/padding_2_2/cpu.vmfb

iree-compile \
  --iree-hal-target-backends=cuda \
  --iree-hal-cuda-llvm-target-arch=sm_70 \
  output/padding_2_2/module.mlir \
  -o output/padding_2_2/cuda.vmfb
