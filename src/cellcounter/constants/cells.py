from enum import Enum

CELL_IDX_NAME = "label"


class CellColumns(Enum):
    COUNT = "count"
    VOLUME = "volume"
    SUM_INTENSITY = "sum_intensity"
    IOV = "iov"


CELL_AGG_MAPPINGS = {
    CellColumns.COUNT.value: "sum",
    CellColumns.VOLUME.value: "sum",
    CellColumns.SUM_INTENSITY.value: "sum",
}
