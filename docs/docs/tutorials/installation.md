# Installation

This guide walks you through installing CellCounter from scratch. Choose the path that matches your setup.

---

## What You'll Need

Before starting, ensure you have:

- [ ] A computer running **Ubuntu 24.04 LTS** (other Linux distributions may work but are untested)
- [ ] **Python 3.12** (this exact version is required)
- [ ] At least **50GB free disk space** (for atlas + sample data)
- [ ] **(Recommended)** An NVIDIA GPU with 8GB+ VRAM for processing large images

---

## Platform-Specific Setup

=== "With GPU (Recommended)"

    If you have an NVIDIA GPU, you'll be able to process whole-brain images (~90GB) efficiently.

    ### Step 1: Install Build Tools

    ```bash
    sudo apt update
    sudo apt install build-essential python3-dev gcc g++
    ```

    ### Step 2: Install NVIDIA Drivers

    ```bash
    # Check available drivers
    sudo ubuntu-drivers list

    # Install the recommended driver (replace with your recommended version)
    sudo ubuntu-drivers install nvidia-driver-590-open

    # Reboot and verify
    sudo reboot
    nvidia-smi
    ```

    ### Step 3: Install CUDA Toolkit

    Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads):

    - **Operating System**: Linux
    - **Architecture**: x86_64
    - **Distribution**: Ubuntu
    - **Version**: 24.04
    - **Installer Type**: deb (local)

    Follow the provided installation commands, then verify:

    ```bash
    nvcc --version
    ```

=== "CPU Only"

    CellCounter can run without a GPU, but processing large images may be slow or run out of memory.

    ### Step 1: Install Build Tools

    ```bash
    sudo apt update
    sudo apt install build-essential python3-dev gcc g++
    ```

    ### Step 2: Verify Installation

    ```bash
    python3 --version  # Should show 3.12.x
    ```

    !!! warning "Performance Warning"
        CPU-only mode is suitable for small test images (<10GB). For whole-brain images (~90GB), a GPU is strongly recommended.

---

## Install CellCounter

### Step 1: Install uv (Python Package Manager)

[uv](https://docs.astral.sh/uv/) is a fast, modern Python package manager.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Step 2: Clone and Install

```bash
# Clone the repository
git clone https://github.com/tlee08/cellcounter.git
cd cellcounter

# Install dependencies
uv sync
```

### Step 3: Install GPU Support (GPU users only)

```bash
uv sync --extra gpu
```

This installs CuPy and Dask-CUDA for GPU acceleration.

---

## Download Reference Atlas

CellCounter needs the Allen Mouse Brain Atlas for registration. Download it with:

```bash
uv run cellcounter-init
```

This will:
- Download the reference brain atlas (~1GB)
- Download the annotation volume (~1GB)
- Download the region hierarchy mapping (~100MB)
- Place them in `~/.cellcounter/atlas/`

The download may take 5-10 minutes depending on your connection.

!!! tip "Atlas Location"
    The atlas is saved in your home directory under `~/.cellcounter/atlas/`. You only need to do this once — all projects will share this atlas.

---

## Verify Installation

Test that everything is working:

```bash
# Test Python imports
uv run python -c "from cellcounter import Pipeline; print('✓ CellCounter installed successfully')"

# Test GPU (if applicable)
uv run python -c "import cupy; print('✓ GPU (CuPy) available')"
```

Expected output:
```
✓ CellCounter installed successfully
✓ GPU (CuPy) available  # (GPU users only)
```

---

## Common Installation Issues

### Issue: "python3-dev not found"

**Solution**: Install the Python development headers:

```bash
sudo apt install python3.12-dev
```

### Issue: "CUDA not found" (GPU users)

**Solution**: Check that CUDA is in your PATH:

```bash
# Add to ~/.bashrc if needed
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then reload:

```bash
source ~/.bashrc
```

### Issue: "No module named cellcounter"

**Solution**: Make sure you're running commands from the cellcounter directory:

```bash
cd /path/to/cellcounter
uv sync
```

---

## Next Steps

Now that CellCounter is installed, proceed to the [Quick Start tutorial](quickstart.md) to process your first image.

Or, if you have multiple images to process, see [Batch Processing](../how-to/batch.md).
