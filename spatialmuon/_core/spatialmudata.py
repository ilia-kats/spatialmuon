from typing import Optional, Dict, Union, Literal
from os import PathLike, path

import h5py

from .backing import BackableObject, BackedDictProxy
from .spatialmodality import SpatialModality
from ..utils import is_h5smu


class SpatialMuData(BackableObject, BackedDictProxy):
    def __init__(
        self,
        backing: Union[str, PathLike, h5py.Group] = None,
        modalities: Optional[Dict[str, SpatialModality]] = None,
        backingmode: Literal["r", "r+"] = "r+",
    ):
        from .. import __version__, __spatialmudataversion__

        if isinstance(backing, PathLike) or isinstance(backing, str):
            assert backingmode in ["r", "r+"], "Argument `backingmode` must be r or r+"

            if path.isfile(backing) and is_h5smu(backing):
                backing = h5py.File(backing, backingmode)
            else:
                f = h5py.File(backing, "w", userblock_size=512, libver="latest")
                f.create_group("mod")
                f.close()
                with open(backing, "br+") as outfile:
                    outfile.write(
                        f"SpatialMuData (format-version={__spatialmudataversion__};creator=spatialmuon;creator-version={__version__})".encode(
                            "utf-8"
                        )
                    )
                backing = h5py.File(backing, "r+")

        super().__init__(backing, key="mod", items=modalities)
        if self.isbacked:
            for m, mod in self.backing["mod"].items():
                self[m] = SpatialModality(backing=mod)
        elif modalities is not None:
            self.update(modalities)

    def _set_backing(self, grp: Union[None, h5py.Group]):
        super()._set_backing(grp)
        if grp is not None:
            self._write_attributes(grp)
            parent = grp.require_group("mod")
            for m, mod in self.items():
                mod.set_backing(parent, m)
        else:
            for mod in self.values():
                mod.set_backing(None)

    @staticmethod
    def _encoding():
        return "SpatialMuData"

    @staticmethod
    def _encodingversion():
        from .. import __spatialmudataversion__

        return __spatialmudataversion__

    def _write_attributes_impl(self, obj: h5py.Group):
        from .. import __version__

        obj.attrs["encoder"] = "spatialmuon"
        obj.attrs["encoder-version"] = __version__

    def _write(self, grp):
        for m, mod in self.items():
            mod.write(grp.require_group("mod"), m)
