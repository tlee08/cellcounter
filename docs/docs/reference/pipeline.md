# Pipeline API

Complete API reference for the `Pipeline` class.

---

## Pipeline

`cellcounter.pipeline.pipeline.Pipeline`

The main orchestrator for cell counting workflows.

### Constructor

```python
Pipeline(proj_dir: str | Path, *, tuning: bool = False)
```

**Parameters**:

| Parameter  | Type            | Description                      |
| ---------- | --------------- | -------------------------------- |
| `proj_dir` | `str` \| `Path` | Path to project directory        |
| `tuning`   | `bool`          | If True, use tuning subdirectory |

**Example**:

```python
from cellcounter import Pipeline

# Main pipeline
pipeline = Pipeline("/path/to/project")

# Tuning pipeline (processes crop only)
pipeline_tuning = Pipeline("/path/to/project", tuning=True)
```

---

## Properties

### `config`

```python
@property
def config(self) -> ProjConfig
```

Access project configuration.

**Returns**: `ProjConfig` — Pydantic model with all settings

**Example**:

```python
print(pipeline.config.cell_counting.threshd_value)  # 60
```

---

### `pfm`

```python
@property
def pfm(self) -> ProjFp
```

Access project filepath model.

**Returns**: `ProjFp` — All file paths in the project

**Example**:

```python
print(pipeline.pfm.cells_agg_csv)
# /path/to/project/cellcount/cells_agg.csv
```

---

### `tuning`

```python
@property
def tuning(self) -> bool
```

Whether this is a tuning instance.

**Returns**: `bool`

---

## Configuration Methods

### `update_config`

```python
def update_config(self, updates) -> None
```

Update project configuration and save to disk.

**Parameters**:

| Parameter | Type | Description               |
| --------- | ---- | ------------------------- |
| `updates` | Any  | Key-value pairs to update |

**Example**:

```python
pipeline.update_config(
    cell_counting={"threshd_value": 45},
    registration={"ref_orientation": {"z": -2, "y": 3, "x": 1}},
)
```

---

### `set_gpu`

```python
def set_gpu(self, *, enabled: bool = True) -> None
```

Switch between GPU and CPU mode.

**Parameters**:

| Parameter | Type   | Default | Description                   |
| --------- | ------ | ------- | ----------------------------- |
| `enabled` | `bool` | `True`  | Use GPU if True, CPU if False |

**Example**:

```python
pipeline.set_gpu(enabled=True)   # Use CuPy
pipeline.set_gpu(enabled=False)  # Use NumPy
```

---

## Input/Output Methods

### `tiff2zarr`

```python
def tiff2zarr(self, in_fp: str | Path, *, overwrite: bool = False) -> None
```

Convert TIFF file(s) to chunked Zarr format.

**Parameters**:

| Parameter   | Type            | Description                             |
| ----------- | --------------- | --------------------------------------- |
| `in_fp`     | `str` \| `Path` | Path to TIFF file or directory of TIFFs |
| `overwrite` | `bool`          | Overwrite existing output               |

**Example**:

```python
# Single multi-page TIFF
pipeline.tiff2zarr("/path/to/brain.tiff")

# Directory of single-plane TIFFs
pipeline.tiff2zarr("/path/to/tiff_folder/")
```

---

### `rechunk_raw`

```python
def rechunk_raw(self) -> None
```

Rechunk raw Zarr array to configured chunk size.

Useful if chunk size needs adjustment after initial conversion.

---

### `clean_proj`

```python
def clean_proj(self) -> None
```

Remove all cellcount subdirectories.

**Warning**: Destructive operation. Deletes all processing outputs.

---

## Registration Pipeline

### `reg_ref_prepare`

```python
def reg_ref_prepare(self, *, overwrite: bool = False) -> None
```

Prepare reference atlas images for registration.

Copies and preprocesses reference/annotation images from atlas, applying orientation and trimming.

**Output files**:

| File          | Description                        |
| ------------- | ---------------------------------- |
| `ref.tiff`    | Reference brain image              |
| `annot.tiff`  | Annotation volume                  |
| `map.csv`     | Region hierarchy mapping           |
| `affine.txt`  | Affine transformation parameters   |
| `bspline.txt` | B-spline transformation parameters |

---

### `reg_img_rough`

```python
def reg_img_rough(self, *, overwrite: bool = False) -> None
```

Rough downsampling by integer strides.

First pass downsampling for registration pyramid.

---

### `reg_img_fine`

```python
def reg_img_fine(self, *, overwrite: bool = False) -> None
```

Fine downsampling using Gaussian zoom.

Second pass downsampling for registration pyramid.

---

### `reg_img_trim`

```python
def reg_img_trim(self, *, overwrite: bool = False) -> None
```

Trim downsampled image to region of interest.

---

### `reg_img_bound`

```python
def reg_img_bound(self, *, overwrite: bool = False) -> None
```

Apply intensity bounds to trimmed image.

Clips intensities for better registration contrast.

---

### `reg_elastix`

```python
def reg_elastix(self, *, overwrite: bool = False) -> None
```

Register image to atlas using elastix.

Performs non-rigid registration to align image to reference atlas.

---

### `make_tuning_arr`

```python
def make_tuning_arr(self, *, overwrite: bool = False) -> None
```

Crop raw Zarr to create tuning subset.

Creates a smaller Zarr for fast parameter tuning.

---

## Cell Counting Pipeline

### `tophat_filter`

```python
def tophat_filter(self, *, overwrite: bool = False) -> None
```

Step 1: Top-hat filter for background removal.

Uses morphological opening to remove large-scale illumination variations.

---

### `dog_filter`

```python
def dog_filter(self, *, overwrite: bool = False) -> None
```

Step 2: Difference of Gaussians for edge enhancement.

Enhances cell boundaries while suppressing slow intensity variations.

---

### `adaptive_threshold_prep`

```python
def adaptive_threshold_prep(self, *, overwrite: bool = False) -> None
```

Step 3: Gaussian subtraction for adaptive thresholding.

Subtracts large-scale Gaussian to normalize local contrast.

---

### `threshold`

```python
def threshold(self, *, overwrite: bool = False) -> None
```

Step 4: Manual intensity thresholding.

Creates binary mask based on `threshd_value` parameter.

---

### `label_thresholded`

```python
def label_thresholded(self, *, overwrite: bool = False) -> None
```

Step 5: Label contiguous regions in thresholded image.

Assigns unique labels to connected foreground components.

---

### `compute_thresholded_volumes`

```python
def compute_thresholded_volumes(self, *, overwrite: bool = False) -> None
```

Step 6: Compute connected component volumes.

Uses union-find to merge labels across chunk boundaries.

---

### `filter_thresholded`

```python
def filter_thresholded(self, *, overwrite: bool = False) -> None
```

Step 7: Filter thresholded objects by size.

Removes objects outside size range (`min_threshd_size`, `max_threshd_size`).

---

### `detect_maxima`

```python
def detect_maxima(self, *, overwrite: bool = False) -> None
```

Step 8: Detect local maxima as cell candidates.

Finds intensity peaks within radius `maxima_radius`.

---

### `label_maxima`

```python
def label_maxima(self, *, overwrite: bool = False) -> None
```

Step 9: Label maxima points.

Assigns unique labels to each detected maximum.

---

### `watershed`

```python
def watershed(self, *, overwrite: bool = False) -> None
```

Step 10: Watershed segmentation.

Separates touching cells using watershed algorithm on maxima labels.

---

### `compute_watershed_volumes`

```python
def compute_watershed_volumes(self, *, overwrite: bool = False) -> None
```

Compute watershed cell volumes.

Uses union-find for cross-chunk volume computation.

---

### `filter_watershed`

```python
def filter_watershed(self, *, overwrite: bool = False) -> None
```

Step 12: Filter watershed objects by size.

Final size filtering using `min_wshed_size` and `max_wshed_size`.

---

### `save_cells_table`

```python
def save_cells_table(self, *, overwrite: bool = False) -> None
```

Step 13: Extract and save cells table.

Extracts cell measurements (coordinates, volume, intensity) and saves to Parquet.

---

## Mapping Pipeline

### `transform_coords`

```python
def transform_coords(self, *, overwrite: bool = False) -> None
```

Transform cell coordinates to reference atlas space.

Applies registration transformation to raw coordinates.

---

### `cell_mapping`

```python
def cell_mapping(self, *, overwrite: bool = False) -> None
```

Map transformed cell coordinates to region IDs.

Assigns each cell to an anatomical brain region using annotation volume.

---

### `group_cells`

```python
def group_cells(self, *, overwrite: bool = False) -> None
```

Group cells by region and aggregate statistics.

Aggregates cell counts, volumes, and intensities per brain region.

---

### `cells2csv`

```python
def cells2csv(self, *, overwrite: bool = False) -> None
```

Save aggregated cell data to CSV.

Exports final results in a readable table format.

---

## Orchestration Methods

### `run_pipeline`

```python
def run_pipeline(
    self,
    in_fp: str,
    *,
    steps: list[str] | None = None,
    overwrite: bool = False
) -> None
```

Run pipeline steps in order.

**Parameters**:

| Parameter   | Type        | Description                     |
| ----------- | ----------- | ------------------------------- |
| `in_fp`     | `str`       | Input file path for `tiff2zarr` |
| `steps`     | `list[str]` | Optional: specific steps to run |
| `overwrite` | `bool`      | Overwrite existing outputs      |

**Example**:

```python
# Run full pipeline
pipeline.run_pipeline("/path/to/image.tiff")

# Run only specific steps
pipeline.run_pipeline(
    "/path/to/image.tiff",
    steps=["tiff2zarr", "tophat_filter", "dog_filter"]
)
```

---

## Class Methods

### `get_imgs_ls`

```python
@staticmethod
def get_imgs_ls(imgs_dir: Path | str) -> list[Path]
```

Get sorted list of subdirectories in a directory.

Useful for batch processing multiple images.

**Parameters**:

| Parameter  | Type            | Description                        |
| ---------- | --------------- | ---------------------------------- |
| `imgs_dir` | `str` \| `Path` | Directory containing image folders |

**Returns**: `list[Path]` — Naturally sorted subdirectory paths

**Example**:

```python
imgs = Pipeline.get_imgs_ls("/path/to/stitched_imgs")
# Returns: [Path("mouse_01"), Path("mouse_02"), ...]
```

---

## Step Registries

Class attributes listing available pipeline steps:

```python
# Registration steps (7 total)
Pipeline.STEPS_REGISTRATION = (
    "tiff2zarr",       # Convert to Zarr
    "reg_ref_prepare", # Prepare reference
    "reg_img_rough",   # Rough downsample
    "reg_img_fine",    # Fine downsample
    "reg_img_trim",    # Trim to ROI
    "reg_img_bound",   # Intensity bounds
    "reg_elastix",     # Register
)

# Cell counting steps (13 total)
Pipeline.STEPS_CELL_COUNTING = (
    "tophat_filter",
    "dog_filter",
    "adaptive_threshold_prep",
    "threshold",
    "label_thresholded",
    "compute_thresholded_volumes",
    "filter_thresholded",
    "detect_maxima",
    "label_maxima",
    "watershed",
    "compute_watershed_volumes",
    "filter_watershed",
    "save_cells_table",
)

# Mapping steps (4 total)
Pipeline.STEPS_MAPPING = (
    "transform_coords",
    "cell_mapping",
    "group_cells",
    "cells2csv",
)
```

---

## Full Example

```python
from cellcounter import Pipeline

# Initialize
pipeline = Pipeline("/path/to/project")

# Configure
pipeline.update_config(
    registration={
        "ref_orientation": {"z": -2, "y": 3, "x": 1},
        "downsample_rough": {"z": 3, "y": 6, "x": 6},
    },
    cell_counting={
        "threshd_value": 60,
        "min_wshed_size": 1,
        "max_wshed_size": 700,
    },
)

# Run everything
pipeline.run_pipeline("/path/to/image.tiff")

# Or step by step with custom logic
pipeline.tiff2zarr("/path/to/image.tiff")
pipeline.reg_ref_prepare()
# ... custom processing ...
pipeline.cells2csv()
```
