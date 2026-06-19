from abc import ABC, abstractmethod
from pathlib import Path


class AbstractFp(ABC):
    """Abstract filepath model."""

    root_dir: Path

    @property
    @abstractmethod
    def subdirs_ls(self) -> list[str]:
        """Return a list of all subdirectories."""
        ...

    def make_subdirs(self) -> None:
        """Make project directories from all subdirs in."""
        for subdir in self.subdirs_ls:
            if subdir is not None:
                (self.root_dir / subdir).mkdir(parents=True, exist_ok=True)

    def export2dict(self) -> dict:
        """Returns a dict of all the FpModel attributes."""
        export_dict = {}
        # For each attribute in the model
        for attr in dir(self):
            # Skipping private attributes
            if attr.startswith("_"):
                continue
            # If the attribute is a str or Path, add it to the export dict
            if isinstance(getattr(self, attr), str | Path):
                export_dict[attr] = getattr(self, attr)
        # Returning
        return export_dict

    @staticmethod
    def raise_not_implemented_err(attr_name: str) -> None:
        """Raise error if attribute is not implemented."""
        err_msg = (
            f"filepath, {attr_name}, is not implemented.\n"
            "Activate this by calling "
            "'set_implement' or explicitly edit this model.",
        )
        raise NotImplementedError(err_msg)
