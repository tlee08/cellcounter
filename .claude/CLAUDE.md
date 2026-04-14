# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# cellcounter

cFos cell counting and region mapping for whole-brain microscopy images.

## Goal

Automated cFos cell counting for neuroscience research. Process whole-brain microscopy images to detect and count cells, then map them to anatomical regions using atlas registration.

## Commands

```bash
# Install (dev uses uv; other users use conda — see README.md)
uv sync
uv sync --extra gpu   # Optional: GPU support (CUDA 13.x)

# Lint
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest

# Entry points
uv run cellcounter-init           # Download/prepare atlas
uv run cellcounter-make-project   # Create project template
```

## Architecture

```
src/cellcounter/
├── pipeline/
│   ├── abstract_pipeline.py  # Base class: GPU/CPU switching, @_check_overwrite
│   ├── pipeline.py           # Orchestrator (registration → cell counting → mapping)
│   └── visual_check.py       # Visual QC tools
├── models/
│   ├── proj_config/          # Pydantic config submodels (registration, cell_counting, etc.)
│   └── fp_models/            # Filepath models; get_proj_fm() factory
├── funcs/
│   ├── cpu_cellc_funcs.py    # CPU cell counting (injectable xp/xdimage backend)
│   ├── gpu_cellc_funcs.py    # GPU wrapper — inherits CPU, wraps listed methods
│   ├── reg_funcs.py          # Image registration helpers
│   ├── map_funcs.py          # Region mapping + get_cells() DataFrame extraction
│   ├── elastix_funcs.py      # Elastix registration wrappers
│   ├── io_funcs.py           # File I/O
│   └── batch_combine_funcs.py # Batch processing utilities
├── constants/                # Enums and constants (coords, cells, masks, annotations, etc.)
├── utils/                    # Dask cluster helpers, logging, union-find, viewer
├── scripts/                  # CLI entry points (init, make_project)
└── templates/                # User-facing script templates (run_pipeline.py, view_img.py)
```

## Data Flow

```
TIFF → Zarr → Registration (elastix) → Cell Detection → Region Mapping → CSV
```

1. **Registration:** `tiff2zarr → reg_ref_prepare → reg_img_rough → reg_img_fine → reg_img_trim → reg_img_bound → reg_elastix`
2. **Cell Counting:** `tophat_filter → dog_filter → adaptive_threshold_prep → threshold → label_thresholded → compute_thresholded_volumes → filter_thresholded → detect_maxima → label_maxima → watershed → compute_watershed_volumes → filter_watershed → save_cells_table`
3. **Mapping:** `transform_coords → cell_mapping → group_cells → cells2csv`

## Key Patterns

- **ProjConfig**: Pydantic v2 model with all parameters. Read/write via JSON. Use `ProjConfig.ensure(fp)` and `config.update(...)`. Access via `pipeline.config`.
- **ProjFp**: Filepath model with cached config. Use `get_proj_fm(proj_dir, tuning=True/False)`.
- **GPU/CPU switching**: `pipeline.set_gpu(enabled=True/False)` at runtime. GPU (cupy) is strongly recommended for large images (~90GB) — CPU mode may OOM or stall.
- **GPU wrapper pattern**: `GpuCellcFuncs(CpuCellcFuncs)` dynamically wraps methods listed in:
  - `_GPU_METHODS`: return cupy arrays → auto-converted to numpy
  - `_GPU_METHODS_NO_CONVERT`: already return numpy internally
- **Injectable backend**: `CpuCellcFuncs(xp=np, xdimage=scipy.ndimage)` — same code path for CPU and GPU.
- **Dask clusters**: `gpu_cluster()`, `heavy_cluster()`, `busy_cluster()` — use as context manager via `cluster_process()`.
- **Overwrite guard**: `@_check_overwrite("attr1", "attr2")` on pipeline methods (defined in `abstract_pipeline.py`) — skips if output exists and `overwrite=False`.
- **Array ops vs DataFrames**: Methods returning arrays stay in `cpu_cellc_funcs.py`; DataFrame-producing logic (e.g. `get_cells`) goes in `map_funcs.py`.

## Conventions

- **Python 3.12**, **uv** for package management (dev)
- **Ruff** for linting (`select = ["ALL"]` with specific ignores in `pyproject.toml`)
- **Pydantic v2** for data validation
- **Google-style docstrings**
- Images stored as chunked Zarr (default 500³ chunks); intermediate results disk-cached via Dask
- Union-Find (`utils/union_find.py`) used for efficient cross-chunk label merging

## Known Issues

- Large images (~90GB) may OOM or stall at `compute_thresholded_volumes` — use GPU mode
- GPU optional dep: `uv sync --extra gpu` installs `cupy-cuda13x` and `dask-cuda`
