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

Other prompt:

```
In @src/cellcounter/pipeline/pipeline.py, look at the save_cells_table function. This function is supposed to, from 3d arrays (of raw intensities, maxima points, watershed region labels, and watershed region sizes), return a dataframe of the x,y,z coordinates of each foreground object (centred on the maxima coordinate) and it's properties (e.g. object volume, object sum intensity). Because we are operating on chunked zarr arrays, we need to correctly offset the z,y,x coord as well. Is this function computing correctly? What are bugs? What are issues? Where can it be improved?
```
