import inspect
import logging
from abc import ABC
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)


class AbstractFp(ABC):
    """Abstract filepath model."""

    root_dir: Path
    subdirs_ls: list[str]

    def copy(self) -> Self:
        """Deepcopy the filepath model instance."""
        # Getting the class constructor parameters
        params_ls = list(dict(inspect.signature(self.__init__).parameters).keys())
        # Constructing an identical model with the corresponding parameter attributes
        return self.__init__(**{param: getattr(self, param) for param in params_ls})

    def make_subdirs(self) -> None:
        """Make project directories from all subdirs in."""
        for subdir in self.subdirs_ls:
            if subdir is not None:
                (self.root_dir / subdir).mkdir(exist_ok=True)

    def export2dict(self) -> dict:
        """Returns a dict of all the FpModel attributes."""
        export_dict = {}
        # For each attribute in the model
        for attr in dir(self):
            # Skipping private attributes
            if attr.startswith("_"):
                continue
            # If the attribute is a str, add it to the export dict
            if isinstance(getattr(self, attr), str):
                export_dict[attr] = getattr(self, attr)
        # Returning
        return export_dict

    @staticmethod
    def raise_not_implemented_err(attr_name: str) -> None:
        """Raise error if attribute is not implemented."""
        raise NotImplementedError(
            (
                "filepath, %s is not implemented.\n"
                "Activate this by calling "
                "'set_implement' or explicitly edit this model.",
            ),
            attr_name,
        )
