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

Install uv (to make a python venv).
Instructions from [here](https://docs.astral.sh/uv/getting-started/installation/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install

```bash
# Install dependencies
uv sync --all-groups
# 
```

## Using

