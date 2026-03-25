# Installation

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- **Recommended**: NVIDIA GPU with CUDA 13.x for large images

## Install

```bash
# Clone the repository
git clone https://github.com/tlee08/cellcounter.git
cd cellcounter

# Install dependencies
uv sync

# Optional: Install GPU support
uv sync --extra gpu
```

## Initialize Atlas

Download and prepare the reference atlas:

```bash
uv run cellcounter-init
```

This creates the atlas directory structure needed for registration.

## Verify Installation

```python
from cellcounter import Pipeline

# Should not raise any errors
print("Installation successful!")
```

## Next Steps

Proceed to the [Quick Start](quickstart.md) tutorial to process your first image.
