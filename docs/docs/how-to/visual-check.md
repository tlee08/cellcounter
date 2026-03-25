# Visual Quality Control

Verify registration and cell detection results visually.

## Overview

```python
from cellcounter import VisualCheck

vc = VisualCheck("/path/to/project", tuning=False)
```

## Registration Check

Combine registration results for visual inspection:

```python
# Overlay registered image with reference
vc.combine_reg(overwrite=True)
```

## Cell Detection Check

### Heatmaps

Generate heatmaps of detected cells:

```python
# Raw coordinate heatmap (original image space)
vc.coords2heatmap_raw(overwrite=True)
vc.combine_heatmap_raw(overwrite=True)

# Transformed coordinate heatmap (atlas space)
vc.coords2heatmap_trfm(overwrite=True)
vc.combine_heatmap_trfm(overwrite=True)
```

### Combined Outputs

```python
# Combine cell counting intermediates for inspection
vc.combine_cellc(overwrite=True)
```

## Viewing Results

Results are saved as TIFF files that can be opened in:

- **napari**: Multi-dimensional viewer (recommended)
- **ImageJ/Fiji**: Standard bioimage analysis
- **Python**: `tifffile.imread()`

```python
import napari
import tifffile

# Open in napari
viewer = napari.Viewer()
viewer.add_image(tifffile.imread("combined_reg.tiff"))
napari.run()
```

## Available Methods

| Method | Output | Description |
|--------|--------|-------------|
| `combine_reg` | `combined_reg.tiff` | Registration overlay |
| `coords2heatmap_raw` | `heatmap_raw.tiff` | Cell density (raw coords) |
| `coords2heatmap_trfm` | `heatmap_trfm.tiff` | Cell density (atlas coords) |
| `combine_heatmap_raw` | `combined_heatmap_raw.tiff` | Heatmap with image |
| `combine_heatmap_trfm` | `combined_heatmap_trfm.tiff` | Heatmap with reference |
| `combine_cellc` | `combined_cellc.tiff` | Cell counting intermediates |
