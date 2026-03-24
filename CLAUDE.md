# cellcounter

cFos cell counting and region mapping for whole-brain microscopy images.

## Architecture

```
src/cellcounter/
├── pipeline/
│   ├── pipeline.py      # Pipeline orchestrator (registration → cell counting → mapping)
│   └── visual_check.py  # Visual QC tools
├── models/
│   ├── proj_config.py   # Pydantic config model (all tunable parameters)
│   └── fp_models/       # Filepath models for project structure
├── funcs/
│   ├── cpu_cellc_funcs.py   # CPU cell counting ops
│   ├── gpu_cellc_funcs.py   # GPU cell counting ops (cupy)
│   ├── reg_funcs.py          # Image registration
│   └── map_funcs.py          # Coordinate mapping to atlas regions
└── gui/                  # Streamlit GUI
```

## Data Flow

```
TIFF → Zarr → Registration (elastix) → Cell Detection → Region Mapping → CSV
```

**Pipeline steps:**
1. **Registration:** `tiff2zarr → reg_ref_prepare → reg_img_rough → reg_img_fine → reg_img_trim → reg_img_bound → reg_elastix`
2. **Cell Counting:** `cellc1-10` (tophat → dog → adaptive threshold → label → filter → watershed)
3. **Mapping:** `transform_coords → cell_mapping → group_cells → cells2csv`

## Key Patterns

- **ProjConfig**: Pydantic model with all parameters. Read/write via JSON. Access via `pipeline.config`.
- **ProjFp**: Filepath model with cached config. Use `get_proj_fm(proj_dir, tuning=True/False)`.
- **GPU-first**: `GpuCellcFuncs` (cupy) extends `CpuCellcFuncs`. Pipeline uses class-level `cellc_funcs` attribute.
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

Working pipeline for cell counting. Active development on:
- Instance-based config injection (in progress)
- Improved testability
