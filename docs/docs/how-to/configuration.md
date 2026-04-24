# Configuration Guide

Complete reference for tuning CellCounter parameters for your specific data.

---

## Accessing Configuration

There are two ways to work with configuration:

### Option 1: Programmatic (Recommended)

```python
from cellcounter import Pipeline

pipeline = Pipeline("/path/to/project")

# Read current values
print(pipeline.config.cell_counting.threshd_value)  # 60

# Update and save
pipeline.update_config(
    cell_counting={"threshd_value": 45}
)
```

### Option 2: Edit config.json Directly

```bash
# Open in your favorite editor
nano /path/to/project/config.json
```

Changes take effect immediately — no restart needed.

---

## Configuration Structure

```python
class ProjConfig:
    # Zarr chunking
    chunks: DimsConfig              # How image is split for processing

    # Tuning crop region
    tuning_trim: DimsSliceConfig    # Small crop for parameter tuning

    # Nested configuration sections
    cluster: ClusterConfig          # Dask parallel computing settings
    reference: ReferenceConfig      # Atlas reference settings
    registration: RegistrationConfig # Image alignment parameters
    cell_counting: CellCountingConfig # Cell detection parameters
    visual_check: VisualCheckConfig  # Visualization settings
```

---

## Chunks Configuration

Control how your image is divided for parallel processing.

### Why Chunks Matter

| Chunk Size | Memory Usage | Speed | Best For |
|------------|--------------|-------|----------|
| Small (100³) | Low | Slower (more overhead) | Limited RAM |
| Medium (500³) | Moderate | Fast | Most users (default) |
| Large (1000³) | High | Faster (less overhead) | High RAM + GPU |

### Example Configurations

```python
# Ultra-large images with GPU (16GB+ VRAM)
chunks={"z": 1000, "y": 1000, "x": 1000}

# Standard whole-brain lightsheet (default, recommended)
chunks={"z": 500, "y": 500, "x": 500}

# Limited RAM / CPU-only (< 32GB RAM)
chunks={"z": 250, "y": 250, "x": 250}

# Very limited RAM (laptop, test images)
chunks={"z": 128, "y": 128, "x": 128}
```

!!! tip "Chunk Size Rules"
    - Must evenly divide your image dimensions OR
    - CellCounter will pad automatically (slight overhead)
    - Larger chunks = fewer files = faster I/O
    - Smaller chunks = less memory per worker

---

## Registration Configuration

Control how your image is aligned to the Allen Atlas.

### Image Orientation (`ref_orientation`)

Maps your image axes to the atlas coordinate system:

```python
registration={
    "ref_orientation": {"z": -2, "y": 3, "x": 1}
}
```

Understanding the values:
- **Positive (1, 2, 3)**: Use axis 1, 2, or 3 without flipping
- **Negative (-1, -2, -3)**: Use axis 1, 2, or 3 AND flip it

The Allen Atlas axes are:
- **Z**: Inferior → Superior
- **Y**: Anterior → Posterior
- **X**: Left → Right

### Common Orientation Patterns

Your image axes depend on your microscope and how the sample was mounted. Check with napari:

```python
import napari
import tifffile

viewer = napari.Viewer()
viewer.add_image(tifffile.imread("your_image.tiff"))
napari.run()
```

Look at voxel coordinates in napari to determine orientation:

| Your Image | Atlas | Orientation Setting |
|------------|-------|---------------------|
| z=0 at bottom | z=0 at bottom | `{"z": 1, ...}` |
| z=0 at top | z=0 at bottom | `{"z": -1, ...}` |
| x=left, y=anterior | standard | `{"z": 1, "y": 2, "x": 3}` |

### Finding Your Orientation

Method 1: Trial and Error
```python
# Try different orientations, check registration result
pipeline.update_config(
    registration={"ref_orientation": {"z": -2, "y": 3, "x": 1}}
)
pipeline.reg_elastix(overwrite=True)

# Visual check
vc.combine_reg()  # Compare trimmed vs ref vs registered
```

Method 2: Compare Dimensions
```python
import tifffile
import numpy as np

img = tifffile.imread("your_image.tiff")
ref = tifffile.imread("~/.cellcounter/atlas/average_template_25.tif")

print(f"Your image: {img.shape}")  # e.g., (300, 2048, 2048)
print(f"Atlas:      {ref.shape}")  # e.g., (528, 320, 456)

# Match roughly: 300 z-slices ≈ 528 atlas z-slices
# So your z maps to atlas z, likely {'z': ...}
```

### Downsampling

Downsample your high-resolution image to match atlas resolution (~25µm).

```python
registration={
    # Integer stride (fast, first pass)
    "downsample_rough": {"z": 3, "y": 6, "x": 6},

    # Float factor (slow, precise, second pass)
    "downsample_fine": {"z": 1.0, "y": 0.6, "x": 0.6},
}
```

Calculate based on your voxel size:

| Your Resolution | Rough | Fine | Result |
|-----------------|-------|------|--------|
| 1 µm (lightsheet) | (2, 4, 4) | (1.0, 0.6, 0.6) | ~25µm |
| 2 µm | (1, 3, 3) | (1.0, 0.8, 0.8) | ~25µm |
| 5 µm | (1, 2, 2) | (1.0, 1.0, 1.0) | ~25µm |

Formula: `your_resolution × rough × fine ≈ 25`

!!! warning "Registration Failing?"
    If elastix fails to converge, try:
    1. Different `downsample_rough` factors
    2. Adjusting `reg_trim` to focus on brain region
    3. Changing `lower_bound` / `upper_bound` intensity clipping

### Intensity Clipping

```python
registration={
    "lower_bound": 100,        # Pixels below this become...
    "lower_bound_mapto": 0,    # ...this value
    "upper_bound": 5000,       # Pixels above this become...
    "upper_bound_mapto": 5000, # ...this value
}
```

Adjust based on your image histogram:
- **lower_bound**: Background level (exclude dark regions)
- **upper_bound**: Saturation level (exclude saturated regions)

---

## Cell Counting Configuration

Control how cells are detected and segmented.

### Detection Sensitivity (`threshd_value`)

Most important parameter! Controls how bright a voxel must be to be considered "cell":

```python
cell_counting={
    "threshd_value": 60  # Default
}
```

| Threshold | Result | When to Use |
|-----------|--------|-------------|
| 20-40 | Many cells detected, more false positives | High SNR, dim cells |
| 50-70 | Balanced (default) | Standard cFos staining |
| 80-120 | Fewer cells, cleaner | Bright staining, want conservative |

!!! tip "Finding the Right Threshold"
    1. Run tuning with default (60)
    2. Visualize with `vc.combine_cellc()`
    3. If too many debris detected → increase threshold
    4. If cells being missed → decrease threshold
    5. Iterate in steps of 10-20

### Background Removal (`tophat_radius`)

Removes large-scale illumination variations:

```python
cell_counting={
    "tophat_radius": 10  # Default, in voxels
}
```

- **Smaller (5-8)**: Better for small cells, uneven background
- **Larger (12-20)**: Better for large cells, smooth background
- **Too large**: Removes actual cell signal

Visual test: After `tophat_filter`, the background should be uniform gray with cells standing out.

### Edge Enhancement (`dog_sigma1`, `dog_sigma2`)

Difference of Gaussians highlights cell-like structures:

```python
cell_counting={
    "dog_sigma1": 1,  # Inner Gaussian (cell center)
    "dog_sigma2": 4,  # Outer Gaussian (background)
}
```

- **sigma1**: Roughly cell radius in voxels (typical cell ~3-5 voxels)
- **sigma2**: 3-4 × sigma1 (captures background around cell)

Adjust based on cell size:
- Small cells (fine resolution): `sigma1=0.5, sigma2=2`
- Medium cells (default): `sigma1=1, sigma2=4`
- Large cells (thick slices): `sigma1=2, sigma2=6`

### Adaptive Thresholding (`large_gauss_radius`)

```python
cell_counting={
    "large_gauss_radius": 101  # Default
}
```

Large Gaussian subtraction for local contrast enhancement. Usually doesn't need adjustment.

### Cell Size Filters

Filter objects by voxel count to remove debris and clusters:

```python
cell_counting={
    # First filter (on thresholded objects)
    "min_threshd_size": 100,   # Minimum connected component
    "max_threshd_size": 9000,  # Maximum (prevents huge clusters)

    # Second filter (on watershed segments)
    "min_wshed_size": 1,       # Minimum final cell size
    "max_wshed_size": 700,     # Maximum final cell size
}
```

Size guidelines (assume 1µm voxels):

| Cell Type | Min Size | Max Size |
|-----------|----------|----------|
| cFos+ nuclei | 10-50 | 300-500 |
| Neurons | 100-200 | 500-1000 |
| Large cells | 200-500 | 1000-2000 |

!!! tip "Finding Size Range"
    1. Look at your raw image in napari
    2. Measure typical cell diameter
    3. Calculate volume: 4/3 × π × r³
    4. Set min/max ~50% around this

### Maxima Detection (`maxima_radius`)

```python
cell_counting={
    "maxima_radius": 10  # Default, in voxels
}
```

Radius for local maxima detection (cell center identification).

- **Smaller**: More sensitive, may split single cells
- **Larger**: Less sensitive, may merge nearby cells
- **Default works for most**: 10 voxels (~10µm)

---

## Cluster Configuration

Control Dask parallel computing resources.

```python
cluster={
    "heavy_n_workers": 1,           # For memory-intensive ops
    "heavy_threads_per_worker": 6,  # Threads per heavy worker
    "busy_n_workers": 4,            # For I/O-bound ops
    "busy_threads_per_worker": 1,   # Threads per busy worker
}
```

### Cluster Types Used

| Cluster | Method | Workers | Use Case |
|---------|--------|---------|----------|
| **GPU** | `gpu_cluster()` | 1 | Cell counting with CuPy |
| **Heavy** | `heavy_cluster()` | 1 | Watershed (high memory) |
| **Busy** | `busy_cluster()` | 4 | I/O, registration, mapping |

### Adjusting for Your Hardware

**High-end workstation (64GB+ RAM, GPU)**:
```python
cluster={
    "heavy_n_workers": 2,
    "heavy_threads_per_worker": 8,
    "busy_n_workers": 8,
}
```

**Laptop / Limited RAM (16GB)**:
```python
cluster={
    "heavy_n_workers": 1,
    "heavy_threads_per_worker": 2,
    "busy_n_workers": 2,
}
```

---

## Tuning Configuration

Define the crop region for parameter tuning.

```python
tuning_trim={
    "z": {"start": 700, "stop": 800, "step": None},
    "y": {"start": 1000, "stop": 3000, "step": None},
    "x": {"start": 1000, "stop": 3000, "step": None},
}
```

Choose a region that:
- Contains cells typical of your data
- Has some background (not all cells)
- Is representative (not edge artifacts)
- Runs quickly (smaller = faster iteration)

Recommended sizes:
- **Z**: 50-200 slices (depending on z-resolution)
- **Y, X**: 1000-4000 pixels each

---

## Visual Check Configuration

Control visualization outputs.

```python
visual_check={
    "heatmap_raw_radius": 5,    # Raw coordinate heatmap sphere size
    "heatmap_trfm_radius": 3,   # Transformed heatmap sphere size
}
```

---

## Configuration Recipes

### Recipe 1: High-Resolution Lightsheet (1µm)

```python
pipeline.update_config(
    chunks={"z": 500, "y": 500, "x": 500},
    registration={
        "ref_orientation": {"z": -2, "y": 3, "x": 1},
        "downsample_rough": {"z": 2, "y": 4, "x": 4},
        "downsample_fine": {"z": 1.0, "y": 0.6, "x": 0.6},
    },
    cell_counting={
        "tophat_radius": 10,
        "dog_sigma1": 1,
        "dog_sigma2": 4,
        "threshd_value": 60,
        "min_wshed_size": 50,
        "max_wshed_size": 500,
    },
)
```

### Recipe 2: Lower Resolution (5µm)

```python
pipeline.update_config(
    registration={
        "downsample_rough": {"z": 1, "y": 2, "x": 2},
        "downsample_fine": {"z": 1.0, "y": 1.0, "x": 1.0},
    },
    cell_counting={
        "tophat_radius": 5,
        "dog_sigma1": 0.5,
        "dog_sigma2": 2,
        "threshd_value": 50,
        "min_wshed_size": 10,
        "max_wshed_size": 200,
    },
)
```

### Recipe 3: Conservative Detection (Fewer False Positives)

```python
pipeline.update_config(
    cell_counting={
        "threshd_value": 80,       # Higher threshold
        "min_wshed_size": 100,     # Larger minimum
        "tophat_radius": 12,       # More background removal
    },
)
```

### Recipe 4: Sensitive Detection (More Cells, Some False Positives)

```python
pipeline.update_config(
    cell_counting={
        "threshd_value": 40,       # Lower threshold
        "min_wshed_size": 10,      # Smaller minimum
        "tophat_radius": 8,        # Less background removal
    },
)
```

---

## Saving and Loading Configurations

### Save for Reuse

```python
# Save to file for reuse across projects
import json

config = pipeline.config
with open("my_settings.json", "w") as f:
    json.dump(config.model_dump(), f, indent=2)
```

### Load Settings

```python
import json
from cellcounter.models.proj_config import ProjConfig

# Load saved settings
with open("my_settings.json") as f:
    settings = json.load(f)

# Apply to new project
new_pipeline = Pipeline("/new/project")
new_pipeline.update_config(**settings)
```
