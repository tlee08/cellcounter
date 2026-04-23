# Visual Quality Control

Verify that registration and cell detection are working correctly by visualizing intermediate and final outputs.

---

## Why Visual QC Matters

Even with good parameters, always verify:

1. **Registration accuracy** — Is your image aligned to the atlas?
2. **Cell detection quality** — Are real cells being found?
3. **False positive rate** — Is debris being counted as cells?
4. **Cell separation** — Are touching cells split correctly?

!!! tip "Golden Rule"
    Visualize your results before trusting the CSV numbers. A few minutes of QC can save hours of reprocessing.

---

## Tools for Visualization

### napari (Recommended)

```bash
# napari is included with CellCounter
uv run napari
```

Or from Python:

```python
import napari
import tifffile

viewer = napari.Viewer()
viewer.add_image(tifffile.imread("your_file.tiff"))
napari.run()
```

### ImageJ/Fiji

Standard bioimage analysis tool. Open TIFF files directly.

### Python (Quick Checks)

```python
import tifffile
import matplotlib.pyplot as plt

img = tifffile.imread("image.tiff")
plt.imshow(img[100])  # View z-slice 100
plt.show()
```

---

## Checking Registration

Registration aligns your image to the Allen Atlas. Verify it's correct!

### Generate Registration Overlay

```python
from cellcounter import VisualCheck

vc = VisualCheck("/path/to/project", tuning=False)
vc.combine_reg()  # Creates combined_reg.tiff
```

This creates a 3-channel TIFF:
- **Channel 0 (Red)**: Your trimmed/bounded image
- **Channel 1 (Green)**: Reference atlas
- **Channel 2 (Blue)**: Registered image

### What to Look For

Open `combined_reg.tiff` in napari:

```python
import napari
import tifffile

img = tifffile.imread("cellcount/combined_reg.tiff")
viewer = napari.Viewer()
viewer.add_image(img, channel_axis=0, name=["trimmed", "reference", "registered"])
napari.run()
```

Tick the visibility (eye icon) of each channel to compare:

| Check | Good | Bad |
|-------|------|-----|
| Brain outline matches | ✓ Contours align | ✗ Brain shape wrong |
| Major structures | ✓ Hippocampus, cortex aligned | ✗ Structures don't overlap |
| Ventricles | ✓ Dark regions align | ✗ Ventricles offset |

### Troubleshooting Registration

**Issue: Registration is flipped**
- Check `ref_orientation` — one or more axes may need negation
- Example: `{"z": -2, "y": 3, "x": 1}` instead of `{"z": 2, ...}`

**Issue: Registration is rotated 90°**
- Swap y and x in orientation: `{"z": 1, "y": 3, "x": 2}`

**Issue: Brain not centered**
- Adjust `reg_trim` to crop to brain region
- Check `downsample_rough` values match your resolution

**Issue: Registration failed/strange warping**
- Adjust intensity bounds (`lower_bound`, `upper_bound`)
- Try different downsampling factors
- Ensure reference atlas downloaded correctly (`cellcounter-init`)

---

## Checking Cell Detection

### Generate Cell Counting Overlay

```python
from cellcounter import VisualCheck

# For tuning crop (faster)
vc_tuning = VisualCheck("/path/to/project", tuning=True)
vc_tuning.combine_cellc()  # Creates cellcount/tuning/combined_cellc.tiff

# For full image (slower, creates trimmed view)
vc_full = VisualCheck("/path/to/project", tuning=False)
vc_full.combine_cellc()    # Creates cellcount/combined_cellc.tiff
```

This creates a 3-channel TIFF:
- **Channel 0 (Gray)**: Raw image (intensity-scaled)
- **Channel 1 (Blue)**: Thresholded mask (filtered objects)
- **Channel 2 (Green)**: Final detected cells

### Interpreting the Overlay

```python
import napari
import tifffile

img = tifffile.imread("cellcount/tuning/combined_cellc.tiff")
viewer = napari.Viewer()
viewer.add_image(img, channel_axis=0,
                 name=["raw", "thresholded", "cells"],
                 colormap=["gray", "blue", "green"])
napari.run()
```

**What each channel shows:**

| Channel | Shows | What to Check |
|---------|-------|---------------|
| Raw | Original fluorescence | Good SNR, no artifacts |
| Thresholded | Objects passing size filters | Reasonable shapes, no holes |
| Cells | Final watershed segments | One segment per cell |

### Sign of Good Detection

✓ **Green regions (cells) align with bright spots in raw  
✓ **Each bright spot has one green segment**
✓ **Green segments are roughly circular/oval**
✓ **No tiny debris included** (check size filter settings)

### Common Cell Detection Problems

#### Problem: Missing Cells

**Symptoms**: Bright spots in raw have no green overlay

**Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| Threshold too high | Lower `threshd_value` (60 → 40) |
| Size filter too large | Decrease `min_wshed_size` |
| Background not removed | Increase `tophat_radius` |
| Cells too dim | Check image intensity scale |

#### Problem: Too Many False Positives

**Symptoms**: Green regions on dim background, debris

**Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| Threshold too low | Raise `threshd_value` (60 → 80) |
| Size filter too small | Increase `min_wshed_size` |
| Debris not filtered | Adjust `min_threshd_size`, `max_threshd_size` |

#### Problem: Touching Cells Merged

**Symptoms**: One green region covers multiple bright spots

**Solutions**:
- Adjust `dog_sigma1` and `dog_sigma2` for better edge detection
- Check `maxima_radius` — may be too large
- Verify watershed is running (check for `wshed_labels` file)

#### Problem: Cells Split Incorrectly

**Symptoms**: Single cell has multiple green regions

**Solutions**:
- Increase `maxima_radius` to avoid over-splitting
- Adjust `dog_sigma1` closer to actual cell radius
- Check if `min_wshed_size` is fragmenting cells

---

## Checking Cell-to-Region Mapping

Verify cells are correctly mapped to brain regions.

### Generate Heatmaps

```python
from cellcounter import VisualCheck

vc = VisualCheck("/path/to/project", tuning=False)

# Raw coordinates heatmap
vc.coords2heatmap_raw()        # cellcount/heatmap_raw.tiff
vc.combine_heatmap_raw()       # cellcount/combined_heatmap_raw.tiff

# Transformed coordinates heatmap
vc.coords2heatmap_trfm()       # cellcount/heatmap_trfm.tiff
vc.combine_heatmap_trfm()      # cellcount/combined_heatmap_trfm.tiff
```

### Interpreting Heatmaps

**Raw Heatmap**: Cell density in original image space
```python
img = tifffile.imread("cellcount/combined_heatmap_raw.tiff")
# Channels: raw image, thresholded, heatmap
```

**Transformed Heatmap**: Cell density in atlas space
```python
img = tifffile.imread("cellcount/combined_heatmap_trfm.tiff")
# Channels: reference, annotation, heatmap
```

### What to Check

1. **Density matches expectation**: High cFos in expected regions?
2. **No off-brain cells**: Cells should fall within brain boundaries
3. **Smooth distribution**: Hotspots should align with anatomical regions

### Troubleshooting Mapping

**Cells outside brain boundary**:
- Registration failed — check `combined_reg.tiff`
- Intensity clipping too aggressive — adjust bounds

**Cells mapped to wrong regions**:
- Registration misalignment — re-run registration
- Coordinate transform error — check `transform_coords` output

---

## Step-by-Step QC Workflow

### After Registration

```python
# 1. Generate and view registration
vc = VisualCheck("/path/to/project", tuning=False)
vc.combine_reg()

# Open in napari and verify
# - Brain outlines match
# - Major structures align
# - No obvious flipping/rotation
```

✓ **If good**: Proceed to cell detection  
✗ **If bad**: Adjust orientation, re-run registration

---

### After Tuning

```python
# 2. Generate cell counting view on tuning crop
vc = VisualCheck("/path/to/project", tuning=True)
vc.combine_cellc()

# Open in napari and verify
# - Cells are detected (green overlay on bright spots)
# - Cells are separated correctly
# - No obvious false positives
```

✓ **If good**: Proceed to full image  
✗ **If bad**: Adjust cell counting parameters, re-run tuning

---

### After Full Pipeline

```python
# 3. Check final result
vc = VisualCheck("/path/to/project", tuning=False)
vc.combine_cellc()  # May be large, trimmed automatically

# 4. Check mapping (transformed space)
vc.combine_heatmap_trfm()

# Open both and verify final outputs
```

✓ **If good**: Analysis complete!  
✗ **If bad**: Identify problem (detection vs registration) and re-run relevant steps

---

## Viewing Pipeline Intermediates

Sometimes you need to check intermediate files. Use the viewer template:

```python
"""View any intermediate files."""
from cellcounter.models.fp_models import get_proj_fp
from cellcounter.utils.viewer import view_images

proj_dir = "/path/to/your/project"
tuning = False

pfm = get_proj_fp(proj_dir, tuning=tuning)

# Pick which files to view
view_images(
    imgs_fp_ls=[
        # Raw and registration
        pfm.raw,
        # pfm.downsmpl1,
        # pfm.downsmpl2,
        # pfm.trimmed,
        # pfm.bounded,
        # pfm.regresult,

        # Cell counting intermediates
        pfm.bgrm,        # After top-hat filter
        pfm.dog,         # After DoG filter
        pfm.threshd,     # After threshold
        pfm.threshd_volumes,  # After size filter
        pfm.maxima,      # Detected maxima
        pfm.wshed_volumes,    # After watershed
        pfm.wshed_filt,  # Final filtered cells
    ],
    trimmer=(
        slice(500, 510),    # Z range
        slice(1500, 3000),  # Y range
        slice(1500, 3000),  # X range
    ),
)
```

This opens all selected files in napari with linked views.

---

## Automated QC Checklist

Use this checklist for each sample:

| Step | Check | Pass Criteria |
|------|-------|---------------|
| **Registration** | `combined_reg.tiff` | Brain outline matches atlas |
| **Detection (tuning)** | `combined_cellc.tiff` (tuning) | >80% of cells detected |
| **Detection (full)** | `combined_cellc.tiff` (full) | Consistent with tuning |
| **Mapping** | `combined_heatmap_trfm.tiff` | Cells in expected regions |
| **Output** | `cells_agg.csv` | Reasonable cell counts |

---

## Exporting QC Images

For publications or documentation, export QC images:

```python
import napari
import tifffile
from napari.utils.io import imsave

# Load data
img = tifffile.imread("cellcount/tuning/combined_cellc.tiff")

# Create viewer
viewer = napari.Viewer()
viewer.add_image(img, channel_axis=0)

# Adjust contrast, zoom to region of interest
viewer.camera.zoom = 2

# Screenshot
screenshot = viewer.screenshot()
imsave("qc_cell_detection.png", screenshot)
```

---

## Tips for Large Images

Full-resolution images are too large to view entirely. Strategies:

### 1. View Crop Regions

```python
import tifffile
import napari

zarr_arr = tifffile.imread("cellcount/raw.zarr")
# Load only a region
region = zarr_arr[500:510, 1500:3000, 1500:3000]

viewer = napari.Viewer()
viewer.add_image(region)
napari.run()
```

### 2. Use Downsampled Versions

```python
# View downsampled registration images
downsmpl = tifffile.imread("cellcount/downsmpl2.tiff")
# Much smaller than raw!
```

### 3. Visual Check Auto-Trims

The `combine_cellc()` method automatically trims for full images:

```python
vc = VisualCheck("/path/to/project", tuning=False)
vc.combine_cellc()
# Reads trim settings from config.visual_check.cellcount_trim
```

Adjust in config if needed:

```python
pipeline.update_config(
    combine={
        "cellcount_trim": {
            "z": {"start": 750, "stop": 760, "step": None},
            "y": {"start": None, "stop": None, "step": None},
            "x": {"start": None, "stop": None, "step": None},
        }
    }
)
```
