from enum import Enum


class RefVersions(Enum):
    """Reference brain template versions."""

    AVERAGE_TEMPLATE_25 = "average_template_25"
    ARA_NISSL_25 = "ara_nissl_25"


class AnnotVersions(Enum):
    """Annotation atlas versions."""

    CCF_2017_25 = "ccf_2017_25"
    CCF_2016_25 = "ccf_2016_25"
    CCF_2015_25 = "ccf_2015_25"


class MapVersions(Enum):
    """Region mapping versions."""

    ABA_ANNOTATIONS = "ABA_annotations"
    CM_ANNOTATIONS = "CM_annotations"
