# cellcounter

## Prerequisites

Use Ubuntu 24.04 LTS.

Install `gcc`, `g++`, and `python3-dev` compilers.

```bash
sudo apt update
sudo apt install build-essential python3-dev
```

Install Nvidia GPU driver

```bash
sudo ubuntu-drivers list
sudo ubunut-drivers install nvidia-driver-590-open
# or choose your preferred version
# Check installation worked:
nvidia-smi
```

Install Nvidia CUDA-toolkit from [here](developer.nvidia.com/cuda-downloads).

Choose: Linux, x86_64, Ubuntu, 24.04, dev (local)

And follow the "CUDA Toolkit Installer" instructions.

```bash
# Follow instructions from "CUDA Toolkit Installer"
# Check installation worked:
nvcc --version
```

## Install

```bash
# Install conda env
conda env create -f conda_env.yaml
# Load atlas
uv run init
```

## Using

Make template

```bash
cd <your/project/folder>
conda activate
cellcounter-make-project
# Edit this file for your analysis
```
