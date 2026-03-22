In @src/cellcounter/pipeline/pipeline.py, I have some steps where I need to get the contiguous size of foreground objects in the array.
Basically:

    (array of foregound objects from cell4) ->
    [get contiguous sizes, like what cellc5 tries to do] ->
    (array, same shape as input, where each voxel represents contiguous size of it's area if it is a foreground voxel, else 0)

For reference, the logic that I want to use is from @example_get_contiguous_size.py . I also want to store the intermediate labels array as like "threshd_label_arr" (make this labelling a separate step). Outline your proposed changes before implementing

---

Some adjustments:

- Use the same size chunks as all the other arrays (no need to rechunk).
- The cls.cellc_funcs.mask2label method should handle labelling each chunk (and has helpful functionality to decide between cpu and gpu). Use this instead of manual scipy.ndimage.label.

Please adjust proposed changes

---

**Yes! `dask-image` provides this functionality.**

## Option: `dask_image.ndmeasure.label`

```python
from dask_image import ndmeasure

# Single function handles cross-chunk connectivity
labeled_arr, num_features = ndmeasure.label(binary_mask_arr, structure=None)
```

**How it works:**

1. Labels each chunk independently with `scipy.ndimage.label`
2. Builds an adjacency graph of labels that touch across chunk boundaries
3. Uses `scipy.sparse.csgraph.connected_components` to find global components
4. Relabels each chunk with globally unique, merged labels

**For sizes, you'd then use:**

```python
# Get unique label counts
label_ids, counts = da.unique(labeled_arr[labeled_arr > 0], return_counts=True)
```

## Comparison

| Aspect               | Your Implementation         | `dask_image.ndmeasure.label`                |
| -------------------- | --------------------------- | ------------------------------------------- |
| Cross-chunk handling | Union-Find (custom)         | Sparse graph + scipy's connected_components |
| Intermediate storage | Saves `threshd_labels.zarr` | Lazy evaluation (no intermediate)           |
| Maturity             | Custom, needs testing       | Battle-tested library                       |
| Dependencies         | Just numpy/scipy            | Requires `dask-image`                       |
| Control              | Full control over process   | Black box                                   |

## Recommendation

If you're comfortable adding `dask-image` as a dependency, it's worth using. The approach is conceptually similar to yours but better tested and integrated with Dask's scheduler.

If you want minimal dependencies or need the intermediate `threshd_labels` for other purposes, your implementation is solid.

**Want me to refactor to use `dask-image` instead?**
