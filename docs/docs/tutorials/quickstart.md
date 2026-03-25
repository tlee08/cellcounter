# Quick Start

Process your first microscopy image from TIFF to cell counts.

## Overview

```mermaid
flowchart LR
    A[TIFF Images] --> B[Registration]
    A --> C[Cell Detection]
    C --> D[Region Mapping]
    B --> D
    D --> E[Results]

    style A fill:#e1f5fe
    style E fill:#c8e6c9
```

## 0. Install this Software

See installation instructions [here](./installation.md)

## 1. Create a Project

```bash
uv run cellcounter-make-project
```

This creates a project directory with the required structure:

```
my_project/
├── config.json          # All pipeline parameters
└── cellcount/           # Processing outputs
    ├── raw.zarr/        # Converted image data
    ├── regresult.tiff   # Registration result
    └── cells_agg.csv    # Final cell counts
```

## 2. Configure Parameters

Edit `config.json` or update programmatically:

```python
from cellcounter import Pipeline

pipeline = Pipeline("/path/to/project")
pipeline.update_config(
    registration={
        "ref_orientation": {"z": -2, "y": 3, "x": 1},
        "downsample_rough": {"z": 3, "y": 6, "x": 6},
    },
    cell_counting={
        "tophat_radius": 10,
        "threshd_value": 60,
        "min_wshed_size": 1,
        "max_wshed_size": 700,
    },
)
```

## 3. Run the Pipeline

```python
from cellcounter import Pipeline

# Initialize
pipeline = Pipeline("/path/to/project")

# Run all steps
pipeline.run_pipeline("/path/to/image.tiff")

# Or run individual steps
pipeline.tiff2zarr("/path/to/image.tiff")
pipeline.reg_ref_prepare()
pipeline.reg_img_rough()
# ... etc
```

## 4. Check Results

Results are saved to `cells_agg.csv`:

| region_id | region_name | cell_count | avg_intensity |
| --------- | ----------- | ---------- | ------------- |
| 1         | isocortex   | 1523       | 245.3         |
| 2         | hippocampus | 892        | 198.7         |

## Pipeline Steps

### Registration (7 steps)

| Step | Method            | Description                  |
| ---- | ----------------- | ---------------------------- |
| 1    | `tiff2zarr`       | Convert TIFF to chunked Zarr |
| 2    | `reg_ref_prepare` | Prepare reference atlas      |
| 3    | `reg_img_rough`   | Integer-stride downsampling  |
| 4    | `reg_img_fine`    | Gaussian zoom downsampling   |
| 5    | `reg_img_trim`    | Trim to region of interest   |
| 6    | `reg_img_bound`   | Apply intensity bounds       |
| 7    | `reg_elastix`     | Elastix registration         |

### Cell Counting

| Step | Method                      | Description                   |
| ---- | --------------------------- | ----------------------------- |
| 1    | tophat_filter               | 1. Background subtraction     |
| 2    | dog_filter                  | 2. Difference of Gaussians    |
| 3    | adaptive_threshold_prep     | 3. Gaussian subtraction       |
| 4    | threshold                   | 4. Manual thresholding        |
| 5    | label_thresholded           | 5. Label contiguous regions   |
| 6    | compute_thresholded_volumes | 6. Union-find volumes         |
| 7    | filter_thresholded          | 7. Size filtering             |
| 8    | detect_maxima               | 8. Local maxima detection     |
| 9    | label_maxima                | 9. Label maxima               |
| 10   | watershed                   | 10. Watershed segmentation    |
| 11   | compute_watershed_volumes   | 11. Watershed volumes         |
| 12   | filter_watershed            | 12. Size filtering            |
| 13   | save_cells_table            | 13. Extract cell measurements |

### Mapping

| Step | Method           | Description              |
| ---- | ---------------- | ------------------------ |
| 1    | transform_coords | Transform to atlas space |
| 2    | cell_mapping     | Assign region IDs        |
| 3    | group_cells      | Aggregate by region      |
| 4    | cells2csv        | Export results           |

## GPU vs CPU Mode

```python
# GPU mode (default, recommended)
pipeline = Pipeline(proj_dir)
pipeline.set_gpu(enabled=True)

# CPU mode (fallback, may OOM on large images)
pipeline.set_gpu(enabled=False)
```

GPU mode uses CuPy for array operations. Falls back automatically if CUDA unavailable.

## Next Steps

- [Configuration Reference](../reference/config.md) - All tunable parameters
- [Visual QC](../how-to/visual-check.md) - Quality control tools
