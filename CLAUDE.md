# cellcounter

cFos cell counting and region mapping for whole-brain microscopy images.

## Goal

Automated cFos cell counting for neuroscience research. Process whole-brain microscopy images to detect and count cells, then map them to anatomical regions using atlas registration.

## Architecture

```
src/cellcounter/
├── pipeline/
│   ├── abstract_pipeline.py  # Base class with GPU/CPU switching
│   ├── pipeline.py           # Pipeline orchestrator (registration → cell counting → mapping)
│   └── visual_check.py       # Visual QC tools
├── models/
│   ├── proj_config.py        # Pydantic config model (all tunable parameters)
│   └── fp_models/            # Filepath models for project structure
├── funcs/
│   ├── cpu_cellc_funcs.py    # CPU cell counting ops
│   ├── gpu_cellc_funcs.py    # GPU cell counting ops (cupy)
│   ├── reg_funcs.py          # Image registration
│   ├── map_funcs.py          # Coordinate mapping to atlas regions
│   ├── elastix_funcs.py      # Elastix registration wrappers
│   ├── io_funcs.py           # File I/O operations
│   ├── viewer_funcs.py       # Napari viewer utilities
│   └── batch_combine_funcs.py # Batch processing utilities
├── constants/                # Enums and constants (coords, cells, masks, etc.)
├── utils/                    # Utilities (dask, logging, union-find)
├── scripts/                  # CLI entry points
├── gui/                      # Streamlit GUI
└── templates/                # User-facing script templates
```

## Data Flow

```
TIFF → Zarr → Registration (elastix) → Cell Detection → Region Mapping → CSV
```

**Pipeline steps:**
1. **Registration:** `tiff2zarr → reg_ref_prepare → reg_img_rough → reg_img_fine → reg_img_trim → reg_img_bound → reg_elastix`
2. **Cell Counting:** `tophat_filter → dog_filter → adaptive_threshold_prep → threshold → label_thresholded → compute_thresholded_volumes → filter_thresholded → detect_maxima → label_maxima → watershed → compute_watershed_volumes → filter_watershed → save_cells_table`
3. **Mapping:** `transform_coords → cell_mapping → group_cells → cells2csv`

## Key Patterns

- **ProjConfig**: Pydantic model with all parameters. Read/write via JSON. Access via `pipeline.config`.
- **ProjFp**: Filepath model with cached config. Use `get_proj_fm(proj_dir, tuning=True/False)`.
- **GPU/CPU switching**: `pipeline.set_gpu(enabled=True/False)` for runtime mode switching. GPU (cupy) is strongly recommended for large images (~90GB) - CPU mode may run out of memory or stall.
- **Dask clusters**: `gpu_cluster()`, `heavy_cluster()`, `busy_cluster()` - context manager via `cluster_process()`.
- **Overwrite guard**: `@check_overwrite("attr1", "attr2")` decorator on pipeline methods.

## Commands

```bash
# Install
uv sync

# Run linting
uv run ruff check .
uv run ruff format .

# Entry points
uv run cellcounter-init          # Load atlas
uv run cellcounter-make-project # Create project template
uv run cellcounter-project-gui   # Launch GUI
```

## Conventions

- **Python 3.12**, **uv** for package management
- **Ruff** for linting (config in pyproject.toml)
- **Pydantic v2** for data validation
- **Google-style docstrings**
- **GPU code**: Inherit from `CpuCellcFuncs`, override with cupy operations

## Current State

Working pipeline for cell counting.

**Known issues:**
- Large images (~90GB) may run out of memory or stall during `compute_thresholded_volumes` step
- GPU mode is strongly recommended for production use

**Completed:**
- Instance-based config injection via `AbstractPipeline`
- Runtime GPU/CPU switching with `set_gpu()`
