# Batch Processing

Process multiple images efficiently using a single script.

---

## When to Use Batch Processing

Use batch processing when you have:

- Multiple animals from the same experiment
- Consistent imaging parameters across samples
- Need for reproducible, automated analysis

!!! tip "Advantages of Batch Processing"
    - Consistent parameters across all samples
    - Overnight/background processing
    - Easy to rerun with different parameters
    - Automatic result aggregation

---

## Data Organization

Organize your data consistently:

```
stitched_imgs/                    # Raw images
├── mouse_01/
│   ├── tile_001.tiff
│   ├── tile_002.tiff
│   └── ...
├── mouse_02/
│   └── ...
└── mouse_03/
    └── ...

analysis_outputs/                 # Results (auto-created)
├── mouse_01/
│   ├── config.json
│   └── cellcount/
├── mouse_02/
└── mouse_03/
```

!!! note "Naming Matters"
    Use descriptive, consistent folder names. These become your sample identifiers in the final aggregated results.

---

## Creating a Batch Script

CellCounter provides a template you can customize. When you run `cellcounter-make-project`, a `run_pipeline.py` file is created.

### Step 1: Generate Template

```bash
cd /path/to/your/analysis
uv run cellcounter-make-project
# Choose any name; we'll edit the generated script
```

### Step 2: Edit the Script

Open `run_pipeline.py` and customize:

```python
from pathlib import Path
from cellcounter import Pipeline, VisualCheck
from cellcounter.funcs.batch_combine_funcs import combine_root

# =========================================
# CHANGE THESE PATHS
stitched_imgs_dir = Path("/path/to/stitched_imgs")
analysis_root_dir = Path("/path/to/analysis_outputs")
overwrite = False  # Set to True to reprocess existing files
# =========================================

# Get list of folders to process
imgs_ls = Pipeline.get_imgs_ls(stitched_imgs_dir)
# Or specify manually:
# imgs_ls = ["mouse_01", "mouse_02"]

print(f"Found {len(imgs_ls)} images to process")
for img_name in imgs_ls:
    print(f"\nProcessing: {img_name}")

    in_fp = stitched_imgs_dir / img_name
    analysis_img_dir = analysis_root_dir / img_name

    try:
        # Create pipeline
        pipeline = Pipeline(analysis_img_dir)

        # =========================================================
        # CONFIGURE PARAMETERS HERE (applied to all images)
        # =========================================================
        pipeline.update_config(
            # Chunks
            chunks={"z": 500, "y": 500, "x": 500},

            # Registration
            registration={
                "ref_orientation": {"z": -2, "y": 3, "x": 1},
                "downsample_rough": {"z": 3, "y": 6, "x": 6},
            },

            # Cell counting (tune these first!)
            cell_counting={
                "tophat_radius": 10,
                "threshd_value": 60,
                "min_wshed_size": 1,
                "max_wshed_size": 700,
            },

            # Tuning crop for parameter validation
            tuning_trim={
                "z": {"start": 700, "stop": 800, "step": None},
                "y": {"start": 1000, "stop": 3000, "step": None},
                "x": {"start": 1000, "stop": 3000, "step": None},
            },
        )

        # Run full pipeline
        pipeline.run_pipeline(str(in_fp), overwrite=overwrite)

        # Run visual checks
        vc = VisualCheck(analysis_img_dir, tuning=False)
        vc.combine_reg(overwrite=overwrite)

        print(f"✓ Completed: {img_name}")

    except Exception as e:
        print(f"✗ Error in {img_name}: {e}")
        # Continue with next image

# Combine all results into single CSV
print("\nCombining results...")
combine_root(analysis_root_dir, analysis_root_dir.parent, overwrite=True)
print("✓ Batch processing complete!")
```

### Step 3: Run the Script

```bash
cd /path/to/your/analysis
uv run python run_pipeline.py
```

---

## Recommended Workflow

For best results with multiple images:

### Phase 1: Tune on Representative Sample

1. Pick one representative image
2. Run tuning workflow to find good parameters
3. Verify results with visual checks
4. Note the working parameters

### Phase 2: Test on 2-3 Images

```python
# Process just a few first
imgs_ls = ["mouse_01", "mouse_02"]

for img_name in imgs_ls:
    # ... processing code
```

Check that:
- Registration looks good for all
- Cell detection works consistently
- No unexpected errors

### Phase 3: Process All Images

```python
# Process all images
imgs_ls = Pipeline.get_imgs_ls(stitched_imgs_dir)
# Or: imgs_ls = ["mouse_01", "mouse_02", "mouse_03", ... all]
```

Run overnight or in background:

```bash
# Run with nohup to survive logout
nohup uv run python run_pipeline.py > batch.log 2>&1 &

# Check progress
tail -f batch.log
```

---

## Step-by-Step Control

For more control over the pipeline, run steps individually:

```python
for img_name in imgs_ls:
    in_fp = stitched_imgs_dir / img_name
    analysis_img_dir = analysis_root_dir / img_name
    pipeline = Pipeline(analysis_img_dir)

    # Registration (run once per image)
    if not (analysis_img_dir / "cellcount" / "regresult.tiff").exists():
        pipeline.tiff2zarr(in_fp)
        pipeline.reg_ref_prepare()
        pipeline.reg_img_rough()
        pipeline.reg_img_fine()
        pipeline.reg_img_trim()
        pipeline.reg_img_bound()
        pipeline.reg_elastix()
        pipeline.make_tuning_arr()
    else:
        print(f"Registration already done for {img_name}")

    # Cell counting (tune=false for full run)
    pipeline = Pipeline(analysis_img_dir, tuning=False)
    pipeline.tophat_filter()
    pipeline.dog_filter()
    pipeline.adaptive_threshold_prep()
    pipeline.threshold()
    # ... (all cell counting steps)
    pipeline.cells2csv()
```

---

## Handling Failures

Some images may fail. Handle gracefully:

```python
import traceback

failed = []
successful = []

for img_name in imgs_ls:
    try:
        # ... processing code ...
        successful.append(img_name)
    except Exception as e:
        print(f"✗ Failed: {img_name}")
        print(traceback.format_exc())
        failed.append((img_name, str(e)))

# Summary
print(f"\n=== SUMMARY ===")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")
for name, error in failed:
    print(f"  - {name}: {error}")

# Retry failed images (after fixing issues)
# imgs_ls = [name for name, _ in failed]
```

---

## Combining Results

After all images are processed, aggregate results:

```python
from cellcounter.funcs.batch_combine_funcs import combine_root

# Combine all cells_agg.csv files
combine_root(
    src_root=analysis_root_dir,      # Where individual results are
    dst_root=analysis_root_dir,      # Where combined result goes
    overwrite=True,
)
```

Output files:
- `combined_cells_agg.csv` — All cells from all samples
- `combined_summary.csv` — Summary statistics per sample

### Understanding Combined Output

`combined_cells_agg.csv`:

| sample_id | region_id | region_name | cell_count | volume | ... |
|-----------|-----------|-------------|------------|--------|-----|
| mouse_01  | 1         | isocortex   | 1523       | 6.2    | ... |
| mouse_01  | 2         | hippocampus | 892        | 3.8    | ... |
| mouse_02  | 1         | isocortex   | 1489       | 6.0    | ... |

Use this for:
- Group comparisons (control vs. treatment)
- Correlation analyses
- Export to statistical software (R, Prism, etc.)

---

## Parallel Processing

⚠️ **Caution**: Running multiple pipelines in parallel can exhaust GPU memory or disk I/O.

### Option 1: Sequential Processing (Safer)

```python
for img_name in imgs_ls:
    pipeline = Pipeline(analysis_root_dir / img_name)
    pipeline.run_pipeline(...)  # One at a time
```

### Option 2: Parallel at Registration Level

Elastix (the registration tool) can use multiple CPU cores:

```python
# In your shell before running
export OMP_NUM_THREADS=8  # Use 8 cores for registration

# Cell counting still uses GPU but one at a time
```

### Option 3: Separate GPU/CPU Instances

If you have multiple GPUs:

```python
import os

# GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pipeline1 = Pipeline(analysis_root_dir / "mouse_01")

# GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
pipeline2 = Pipeline(analysis_root_dir / "mouse_02")
```

Note: Requires careful management — recommended for advanced users only.

---

## Monitoring Progress

For long-running batches, monitor progress:

```python
import time
from datetime import datetime

total = len(imgs_ls)
start_time = time.time()

for i, img_name in enumerate(imgs_ls, 1):
    print(f"\n[{i}/{total}] {img_name} at {datetime.now()}")

    # ... processing ...

    elapsed = time.time() - start_time
    avg_time = elapsed / i
    remaining = avg_time * (total - i)
    print(f"  ETA: {remaining/60:.1f} minutes remaining")
```

---

## Best Practices

### 1. Start Small

```python
# Test with 1-2 images first
imgs_ls = ["mouse_01"]  # Just one
```

### 2. Use overwrite=False Initially

```python
overwrite = False  # Skip existing files, safe for resuming
```

### 3. Save Configs with Results

```python
# Config is auto-saved, but also document your choices
with open(analysis_root_dir / "processing_notes.txt", "w") as f:
    f.write("Threshold: 60 (determined from mouse_01 tuning)\n")
    f.write("Date: 2024-01-15\n")
    f.write("Operator: Your Name\n")
```

### 4. Verify Periodically

```python
# After every N images, check one result
if i % 5 == 0:
    vc = VisualCheck(analysis_img_dir, tuning=False)
    vc.combine_cellc()
    print("  Visual check saved — verify before continuing!")
```

---

## Example: Complete Batch Script

```python
"""Complete batch processing example."""
from pathlib import Path
import traceback
from cellcounter import Pipeline, VisualCheck
from cellcounter.funcs.batch_combine_funcs import combine_root

def main():
    # Configuration
    stitched_imgs_dir = Path("/data/lab/stitched_imgs")
    analysis_root_dir = Path("/data/lab/analysis")
    overwrite = False

    # Get images (or specify list)
    imgs_ls = Pipeline.get_imgs_ls(stitched_imgs_dir)
    print(f"Found {len(imgs_ls)} images to process")

    # Track stats
    successful = []
    failed = []

    for i, img_name in enumerate(imgs_ls, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(imgs_ls)}] Processing: {img_name}")
        print(f"{'='*50}")

        in_fp = stitched_imgs_dir / img_name
        analysis_img_dir = analysis_root_dir / img_name

        try:
            pipeline = Pipeline(analysis_img_dir)

            # Apply standard configuration
            pipeline.update_config(
                chunks={"z": 500, "y": 500, "x": 500},
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
            pipeline.run_pipeline(str(in_fp), overwrite=overwrite)

            # Visual QC
            vc = VisualCheck(analysis_img_dir, tuning=False)
            vc.combine_reg(overwrite=overwrite)
            vc.combine_cellc(overwrite=overwrite)

            print(f"✓ SUCCESS: {img_name}")
            successful.append(img_name)

        except Exception as e:
            print(f"✗ FAILED: {img_name}")
            print(traceback.format_exc())
            failed.append((img_name, str(e)))

    # Combine results
    if successful:
        print("\n Combining results...")
        combine_root(analysis_root_dir, analysis_root_dir, overwrite=True)

    # Final report
    print(f"\n{'='*50}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Successful: {len(successful)}/{len(imgs_ls)}")
    print(f"Failed: {len(failed)}/{len(imgs_ls)}")

    if failed:
        print("\nFailed images:")
        for name, error in failed:
            print(f"  - {name}: {error}")

if __name__ == "__main__":
    main()
```

Save as `batch_process.py` and run:

```bash
uv run python batch_process.py
```
