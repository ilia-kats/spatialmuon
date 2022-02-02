import warnings
from typing import Optional, Dict, Union, Literal
from os import PathLike, path, system

import h5py

from .backing import BackableObject
from .spatialmodality import SpatialModality
from ..utils import is_h5smu
from .io import repack_h5smu


class SpatialMuData(BackableObject):
    def __init__(
        self,
        backing: Union[str, PathLike, h5py.Group] = None,
        modalities: Optional[Dict[str, SpatialModality]] = None,
        backingmode: Literal["r", "r+", "w"] = "r+",
    ):
        from .. import __version__, __spatialmudataversion__

        if isinstance(backing, PathLike) or isinstance(backing, str):
            assert backingmode in ["r", "r+", "w"], "Argument `backingmode` must be r, r+ or w"
            if path.isfile(backing) and backingmode != "w":
                if is_h5smu(backing):
                    try:
                        backing = h5py.File(backing, backingmode)
                    except OSError as e:
                        if (
                            str(e)
                            == "Unable to open file (file is already open for write/SWMR write (may use <h5clear "
                            "file> to clear file consistency flags))"
                        ):
                            cmd = f"h5clear -s {backing}"
                            system(cmd)
                            backing = h5py.File(backing, backingmode)
                        else:
                            raise e
                else:
                    raise FileExistsError("file already exists and is not a valid SpatialMuon file")
            else:
                # print("creating a new file for the backed storage")
                f = h5py.File(
                    backing,
                    "w",
                    userblock_size=4096,
                    libver="latest",
                    fs_strategy="page",
                    fs_persist=True,
                )
                f.close()
                with open(backing, "br+") as outfile:
                    fname = "SpatialMuData (format-version={};".format(__spatialmudataversion__)
                    fname += "creator=spatialmuon;"
                    fname += "creator-version={})".format(__version__)
                    outfile.write(fname.encode("utf-8"))
                backing = h5py.File(backing, "r+")

        super().__init__(backing, items=modalities)
        if self.is_backed:
            for m, mod in self.backing.items():
                self._set_kv(m, SpatialModality(backing=mod))
                # self[m] = SpatialModality(backing=mod)
        elif modalities is not None:
            warnings.warn("code not tested")
            self.update(modalities)

    @staticmethod
    def _encodingtype():
        return "SpatialMuData"

    @staticmethod
    def _encodingversion():
        from .. import __spatialmudataversion__

        return __spatialmudataversion__

    def _write_impl(self, grp):
        pass

    def _write_attributes_impl(self, obj: h5py.Group):
        from .. import __version__

        obj.attrs["encoder"] = "spatialmuon"
        obj.attrs["encoder-version"] = __version__

    def __repr__(self):
        repr_str = "SpatialMuData object\n"
        for m, mod in self.items():
            repr_str += f"| '{m}'\n"
            for k, v in mod.items():
                repr_str += f"├── '{k}': {v}\n"
        repr_str = repr_str.rstrip("\n")
        repr_str = "\n└──".join(repr_str.rsplit("\n├──", 1))
        return repr_str

    def repack(self, compression_level: Optional[int] = None):
        assert self.is_backed
        f = self.backing.filename
        self.backing.close()
        repack_h5smu(f, compression_level=compression_level)
