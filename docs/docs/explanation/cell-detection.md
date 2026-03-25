# Cell Detection Algorithm

How cFos cells are detected and counted.

## Overview

The cell counting pipeline uses a combination of filtering, thresholding, and watershed segmentation:

```mermaid
flowchart LR
    A[Raw Image] --> B[Top-hat]
    B --> C[DoG]
    C --> D[Adaptive Thresh]
    D --> E[Threshold]
    E --> F[Filter by Size]
    F --> G[Detect Maxima]
    G --> H[Watershed]
    H --> I[Filter by Size]
    I --> J[Cell Table]
```

## Step-by-Step

### 1. Background Removal (Top-hat)

Removes large-scale intensity variations:

```python
# Morphological top-hat filter
# radius = 10 voxels (default)
tophat_filt(image, radius=10)
```

### 2. Edge Enhancement (DoG)

Difference of Gaussians highlights cell-like structures:

```python
# dog_sigma1 = 1, dog_sigma2 = 4 (default)
dog_filt(image, sigma1=1, sigma2=4)
```

### 3. Adaptive Thresholding

Subtracts large-scale background for local thresholding:

```python
# large_gauss_radius = 101 (default)
gauss_subt_filt(image, sigma=101)
```

### 4. Manual Thresholding

Binary mask creation:

```python
# threshd_value = 60 (default)
manual_thresh(image, val=60)
```

### 5. Size Filtering (Thresholded)

Removes objects outside size range:

```python
# min_threshd_size = 100, max_threshd_size = 9000
volume_filter(labels, smin=100, smax=9000)
```

### 6. Maxima Detection

Finds local intensity peaks as cell candidates:

```python
# maxima_radius = 10 (default)
get_local_maxima(image, radius=10, mask=filtered_mask)
```

### 7. Watershed Segmentation

Separates touching cells:

```python
wshed_segm(image, maxima_labels, mask=filtered_mask)
```

### 8. Size Filtering (Watershed)

Final size filtering on segmented objects:

```python
# min_wshed_size = 1, max_wshed_size = 700
volume_filter(wshed_labels, smin=1, smax=700)
```

## Cross-Chunk Processing

Labels must be connected across chunk boundaries using Union-Find:

```python
def _spatial_connect_count(self, label_arr):
    # 1. Find boundary pairs between chunks
    # 2. Union labels that are contiguous
    # 3. Compute volumes for connected components
    # 4. Map volumes back to array
```

This ensures cells spanning chunk boundaries are counted correctly.

## Output Columns

| Column | Description |
|--------|-------------|
| `z`, `y`, `x` | Raw coordinates |
| `z_trfm`, `y_trfm`, `x_trfm` | Transformed coordinates |
| `label` | Watershed label |
| `volume` | Cell volume (voxels) |
| `sum_intensity` | Total fluorescence |
| `mean_intensity` | Average fluorescence |
| `region_id` | Brain region ID |
| `region_name` | Brain region name |
