"""
Connected Components Size Map
==============================
Input  : a uint16 dask array (foreground > 0, background == 0)
Output : a zarr array of the same shape (int64):
           - 0  → background
           - N  → voxel belongs to a contiguous foreground region of N voxels

Pipeline
--------
  1. Rechunk to small, uniform chunks
  2. Label each chunk independently with scipy, using a globally-unique label offset
  3. Save labeled array to zarr (so we can do two cheap passes later)
  4. Overlap labeled array by depth=1 to expose chunk boundaries
  5. For every overlapped chunk, find adjacent label pairs that cross boundaries
  6. Union-Find: merge labels that belong to the same physical region
  7. Count voxels per label → aggregate by merged root → build label→size table
  8. Second pass over labeled zarr: replace each label with its component size
  9. Save output to zarr
"""

import os
from collections import defaultdict

import dask
import dask.array as da
import numpy as np
import zarr
from dask.distributed import Client, LocalCluster
from scipy import ndimage

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_PATH = "input.zarr"  # adjust to your source
LABELED_PATH = "labeled.zarr"  # intermediate: unique-labeled chunks
OUTPUT_PATH = "component_sizes.zarr"  # final output

CHUNK = (200, 200, 200)  # rechunk to this before processing
# each chunk ≈ 16 MB as uint16, 64 MB as int64

# ── CLUSTER ───────────────────────────────────────────────────────────────────

cluster = LocalCluster(
    n_workers=1,
    threads_per_worker=4,  # I/O-bound steps benefit from threads
    memory_limit=0,  # no artificial cap — let the OS manage RAM
)
client = Client(cluster)
print(f"Dashboard: {client.dashboard_link}")


# ── 1. LOAD + RECHUNK ─────────────────────────────────────────────────────────

arr = da.from_zarr(INPUT_PATH)
print(f"Input shape: {arr.shape}  dtype: {arr.dtype}")

arr_rc = arr.rechunk(CHUNK)


# ── 2. LABEL EACH CHUNK WITH GLOBALLY UNIQUE LABELS ──────────────────────────
#
# Why unique labels?
#   scipy.ndimage.label always starts at 1 in every chunk.  If we left labels
#   as-is, chunk A's region "1" and chunk B's completely different region "1"
#   would look identical to the union-find, causing false merges.
#
# Offset strategy:
#   For a chunk of 200³ = 8 000 000 voxels, the theoretical maximum number of
#   connected components is ⌈voxels/2⌉ ≈ 4 000 000 (a perfect checkerboard).
#   We space offsets by (total voxels in chunk + 1) = 8 000 001, so offsets
#   for chunk index k start at k * 8 000 001.  This guarantees no overlap.
#
#   block_info[0]["chunk-location"] gives the n-D chunk index, e.g. (2, 5, 3).
#   block_info[0]["num-chunks"]     gives the grid shape,    e.g. (6, 40, 21).
#   np.ravel_multi_index converts those into a flat integer (the chunk's rank).

MAX_LABELS_PER_CHUNK = int(np.prod(CHUNK)) + 1  # 8_000_001


def label_chunk(block, block_info=None):
    labeled, _ = ndimage.label(block)  # labels 1..K inside chunk
    if block_info and block_info[0]:
        loc = block_info[0]["chunk-location"]  # e.g. (2, 5, 3)
        grid_shape = block_info[0]["num-chunks"]  # e.g. (6, 40, 21)
        flat_idx = int(np.ravel_multi_index(loc, grid_shape))
        offset = flat_idx * MAX_LABELS_PER_CHUNK
        labeled[labeled > 0] += offset  # shift labels to unique range
    return labeled.astype(np.int64)


labeled_dask = arr_rc.map_blocks(label_chunk, dtype=np.int64)


# ── 3. SAVE LABELED ARRAY (intermediate) ──────────────────────────────────────
#
# We write this once now so that steps 5 and 8 can each do a cheap sequential
# read without recomputing the labels from scratch.

print("Saving labeled array …")
labeled_store = zarr.open(
    LABELED_PATH,
    mode="w",
    shape=labeled_dask.shape,
    chunks=CHUNK,
    dtype=np.int64,
)
da.store(labeled_dask, labeled_store, lock=False)
print("Labeled array saved.")

labeled = da.from_zarr(LABELED_PATH)


# ── 4. OVERLAP labeled array by depth=1 ───────────────────────────────────────
#
# depth=1 means each chunk gets a 1-voxel halo from its neighbours on each
# face.  That is sufficient to detect all 6-connected cross-boundary pairs.
# The halo uses 0-fill (boundary="constant") so background voxels don't
# accidentally form pairs with real labels.

labeled_overlap = da.overlap.overlap(labeled, depth=1, boundary=0)


# ── 5. FIND CROSS-BOUNDARY ADJACENT LABEL PAIRS ───────────────────────────────
#
# For each overlapped chunk we compare every voxel with its +1 neighbour along
# each axis.  Because scipy already merged all *intra*-chunk adjacencies, any
# pair (a, b) with a≠b and both>0 must straddle a chunk boundary.
#
# We normalise pairs so min(a,b) < max(a,b) before inserting into a set;
# this deduplicates (a,b) and (b,a) cheaply.


def find_boundary_pairs(block):
    pairs = set()
    for axis in range(block.ndim):
        sl_a = [slice(None)] * block.ndim
        sl_b = [slice(None)] * block.ndim
        sl_a[axis] = slice(None, -1)
        sl_b[axis] = slice(1, None)
        a = block[tuple(sl_a)].ravel()
        b = block[tuple(sl_b)].ravel()
        mask = (a > 0) & (b > 0) & (a != b)
        if mask.any():
            lo = np.minimum(a[mask], b[mask])
            hi = np.maximum(a[mask], b[mask])
            pairs.update(zip(lo.tolist(), hi.tolist()))
    if pairs:
        return np.array(list(pairs), dtype=np.int64)
    return np.empty((0, 2), dtype=np.int64)


print("Finding cross-boundary pairs …")
delayed_blocks = labeled_overlap.to_delayed().ravel()
pair_arrays = dask.compute(
    *[dask.delayed(find_boundary_pairs)(b) for b in delayed_blocks]
)
valid = [p for p in pair_arrays if len(p) > 0]
all_pairs = np.concatenate(valid, axis=0) if valid else np.empty((0, 2), dtype=np.int64)
print(f"Cross-boundary pairs found: {len(all_pairs):,}")


# ── 6. UNION-FIND: merge touching labels ──────────────────────────────────────
#
# Classic path-compressed, rank-unioned disjoint-set structure.
# After this step, find(label) returns the canonical root for the physical
# connected component that label belongs to.

parent = {}
rank = {}


def find(x):
    # Iterative find with path halving
    while parent.get(x, x) != x:
        parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
        x = parent[x]
    return x


def union(x, y):
    px, py = find(x), find(y)
    if px == py:
        return
    rx, ry = rank.get(px, 0), rank.get(py, 0)
    if rx < ry:
        px, py = py, px
    parent[py] = px
    if rx == ry:
        rank[px] = rx + 1


print("Running Union-Find …")
for a, b in all_pairs:
    union(int(a), int(b))
print("Union-Find complete.")


# ── 7. COUNT VOXELS PER LABEL → AGGREGATE BY COMPONENT ROOT ──────────────────
#
# One sequential pass over the labeled zarr (chunk by chunk, never loading the
# whole thing).  We accumulate raw per-label counts, then fold them into
# component totals using the union-find.

print("Counting voxels per label …")
label_voxel_counts: dict[int, int] = defaultdict(int)

labeled_for_count = da.from_zarr(LABELED_PATH)
for blk in labeled_for_count.to_delayed().ravel():
    arr_blk = blk.compute()
    fg = arr_blk.ravel()
    fg = fg[fg > 0]
    if len(fg) == 0:
        continue
    unique, counts = np.unique(fg, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        label_voxel_counts[u] += c

print(f"Unique labels (foreground): {len(label_voxel_counts):,}")

# Aggregate individual label counts into component totals
component_sizes: dict[int, int] = defaultdict(int)
for label, count in label_voxel_counts.items():
    component_sizes[find(label)] += count

n_components = len(component_sizes)
max_size = max(component_sizes.values()) if component_sizes else 0
print(f"Connected components: {n_components:,}   largest: {max_size:,} voxels")

# Build a fast lookup: label → component size
# We use sorted numpy arrays so each chunk can use vectorised searchsorted
# instead of a slow per-voxel Python dict lookup.
label_keys = np.array(list(label_voxel_counts.keys()), dtype=np.int64)
label_sizes = np.array(
    [component_sizes[find(lbl)] for lbl in label_keys], dtype=np.int64
)
sort_idx = np.argsort(label_keys)
sorted_keys = label_keys[sort_idx]
sorted_sizes = label_sizes[sort_idx]


# ── 8. WRITE OUTPUT ARRAY ─────────────────────────────────────────────────────
#
# Second pass over labeled zarr.  For each voxel:
#   - background (label == 0) → 0
#   - foreground              → size of its connected component
#
# The lookup uses numpy searchsorted for fully vectorised, no-Python-loop
# performance per chunk.


def map_labels_to_sizes(block):
    out = np.zeros(block.shape, dtype=np.int64)
    mask = block > 0
    if not mask.any():
        return out
    labels_fg = block[mask]
    # Find each label's position in the sorted lookup table
    pos = np.searchsorted(sorted_keys, labels_fg)
    pos = np.clip(pos, 0, len(sorted_keys) - 1)
    valid = sorted_keys[pos] == labels_fg  # guard against missing keys
    out[mask] = np.where(valid, sorted_sizes[pos], 0)
    return out


print("Writing output array …")
labeled_for_output = da.from_zarr(LABELED_PATH)
output_dask = labeled_for_output.map_blocks(map_labels_to_sizes, dtype=np.int64)

output_store = zarr.open(
    OUTPUT_PATH,
    mode="w",
    shape=output_dask.shape,
    chunks=CHUNK,
    dtype=np.int64,
)
da.store(output_dask, output_store, lock=False)
print(f"Done!  Output saved to '{OUTPUT_PATH}'")


# ── 9. OPTIONAL: APPLY SIZE FILTER IN-PLACE ───────────────────────────────────
#
# If you want to zero out regions that are too small or too large immediately:
#
#   MIN_SIZE = 10_000       # voxels — tune to your data
#   MAX_SIZE = 50_000_000   # voxels — tune to your data
#
#   result = da.from_zarr(OUTPUT_PATH)
#   filtered = da.where((result >= MIN_SIZE) & (result <= MAX_SIZE), result, 0)
#
#   filtered_store = zarr.open(
#       "filtered.zarr", mode="w",
#       shape=filtered.shape, chunks=CHUNK, dtype=np.int64,
#   )
#   da.store(filtered, filtered_store, lock=False)
#
# Or you can do this lazily during downstream processing — load the zarr, apply
# the threshold, and never write a separate file at all.


# ── CLEANUP ───────────────────────────────────────────────────────────────────
#
# The labeled.zarr intermediate is only needed for the two passes above.
# Once output is verified you can remove it:
#
#   import shutil
#   shutil.rmtree(LABELED_PATH)
