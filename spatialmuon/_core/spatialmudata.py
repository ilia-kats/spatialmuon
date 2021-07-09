from typing import Optional, Dict, Union

import h5py

from .backing import BackableObject
from .spatialmodality import SpatialModality

class SpatialMuData(BackableObject, dict):
    def __init__(self, modalities:Optional[Dict[str, SpatialModality]]=None, backing: h5py.Group = None):
        super().__init__(backing)
        if self.isbacked:
            for m, mod in self.backing["mod"].items():
                super().__setitem__(m, SpatialModality(backing=mod))
        elif modalities is not None:
            self.update(modalities)

    def _set_backing(self, grp:Union[None, h5py.Group]):
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

    def __setitem__(self, key: str, mod:SpatialModality):
        super().__setitem__(key, mod)
        if self.isbacked:
            mod.backing = self.backing.require_group(f"mod/{key}")
