# Reproducer for IREE Bug

## Installation

- Create and activate a virtual environemt with python `3.10`
- Get your CUDA version with `nvidia-smi`
- for CUDA 11 run `pip install -r requirements-torch-cu118.txt`
- for CUDA 12 run `pip install -r requirements-torch-cu121.txt`
- run `pip install -r requirements-iree.txt`
- run `pip install -r requirements-shark-turbine.txt`
- run `./compile.sh` (this should give no output)
- run `./run.sh` (this should output `EXEC @main` several times)
- run `./compares.sh` (this should output `Max difference: xxx` several times)
