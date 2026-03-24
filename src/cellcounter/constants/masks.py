from enum import Enum

MASK_VOLUME = "volume"


class MaskColumns(Enum):
    VOLUME_ANNOT = f"{MASK_VOLUME}_annot"
    VOLUME_MASK = f"{MASK_VOLUME}_mask"
    VOLUME_PROP = f"{MASK_VOLUME}_prop"
