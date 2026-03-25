# Batch Processing

Process multiple images in a single run.

## Directory Structure

Organize your data as follows:

```
stitched_imgs/
├── mouse_01/
│   ├── tile_001.tiff
│   ├── tile_002.tiff
│   └── ...
├── mouse_02/
│   └── ...
└── mouse_03/

analysis_outputs/
├── mouse_01/
│   ├── config.json
│   └── cellcount/
├── mouse_02/
└── mouse_03/
```

## Batch Script

Create a script based on the template:

```python
from pathlib import Path
from cellcounter import Pipeline, VisualCheck
from cellcounter.funcs.batch_combine_funcs import combine_root

# Configuration
stitched_imgs_dir = Path("/path/to/stitched_imgs")
analysis_root_dir = Path("/path/to/analysis_outputs")
overwrite = False

# Get list of images to process
imgs_ls = Pipeline.get_imgs_ls(stitched_imgs_dir)
# Or specify manually:
# imgs_ls = ["mouse_01", "mouse_02"]

for img_name in imgs_ls:
    print(f"Processing: {img_name}")

    in_fp = stitched_imgs_dir / img_name
    analysis_img_dir = analysis_root_dir / img_name

    # Create pipeline
    pipeline = Pipeline(analysis_img_dir)

    # Update config (first run only)
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

    # Run pipeline
    try:
        pipeline.run_pipeline(str(in_fp), overwrite=overwrite)
    except Exception as e:
        print(f"Error in {img_name}: {e}")

# Combine results across all experiments
combine_root(analysis_root_dir, analysis_root_dir.parent)
```

## Step-by-Step Control

For more control over individual steps:

```python
for img_name in imgs_ls:
    pipeline = Pipeline(analysis_root_dir / img_name)

    # Registration (run once)
    pipeline.tiff2zarr(in_fp, overwrite=False)
    pipeline.reg_ref_prepare(overwrite=False)
    pipeline.reg_img_rough(overwrite=False)
    pipeline.reg_img_fine(overwrite=True)      # Tuning
    pipeline.reg_img_trim(overwrite=True)      # Tuning
    pipeline.reg_img_bound(overwrite=True)     # Tuning
    pipeline.reg_elastix(overwrite=True)       # Tuning

    # Cell counting (iterate on tuning first)
    for is_tuning in [True, False]:
        p = Pipeline(analysis_root_dir / img_name, tuning=is_tuning)
        p.tophat_filter(overwrite=overwrite)
        # ... other steps
```

## Tuning Workflow

1. **Run registration once** - Registration parameters are shared
2. **Tune cell counting** - Use `tuning=True` with small crop
3. **Run full pipeline** - Once parameters are validated

```python
# Tuning mode processes a small crop (defined in config.tuning_trim)
pipeline_tuning = Pipeline(proj_dir, tuning=True)
pipeline_tuning.run_pipeline(in_fp)

# Full mode processes the entire image
pipeline_full = Pipeline(proj_dir, tuning=False)
pipeline_full.run_pipeline(in_fp)
```
