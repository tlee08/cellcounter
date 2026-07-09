CELL_IDX_NAME = "label"

# ===============================================
# Cell Columns
# ===============================================


COUNT = "count"
VOLUME = "volume"
SUM_INTENSITY = "sum_intensity"
IOV = "iov"

CELL_COLUMNS = [COUNT, VOLUME, SUM_INTENSITY, IOV]


CELL_AGG_MAPPINGS = {
    COUNT: "sum",
    VOLUME: "sum",
    SUM_INTENSITY: "sum",
}
