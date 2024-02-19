#!/bin/bash

iree-run-module \
  --device=local-task \
  --module=output/turbine/padding_1_1/cpu.vmfb \
  --function=main \
  --input=@output/module_input.npy \
  --output=@output/turbine/padding_1_1/cpu_output.npy

iree-run-module \
  --device=cuda \
  --module=output/turbine/padding_1_1/cuda.vmfb \
  --function=main \
  --input=@output/module_input.npy \
  --output=@output/turbine/padding_1_1/cuda_output.npy

iree-run-module \
  --device=local-task \
  --module=output/turbine/padding_2_2/cpu.vmfb \
  --function=main \
  --input=@output/module_input.npy \
  --output=@output/turbine/padding_2_2/cpu_output.npy

iree-run-module \
  --device=cuda \
  --module=output/turbine/padding_2_2/cuda.vmfb \
  --function=main \
  --input=@output/module_input.npy \
  --output=@output/turbine/padding_2_2/cuda_output.npy
