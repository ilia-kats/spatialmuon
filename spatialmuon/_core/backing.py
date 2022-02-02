import os
from abc import ABC, abstractmethod
from collections import UserDict
from typing import Optional, Union, Callable
import warnings

import h5py


class BackableObject(ABC):
    def __init__(self, backing: Optional[Union[h5py.Group, h5py.Dataset]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backing = backing

    @staticmethod
    @abstractmethod
    def _encodingtype() -> str:
        pass

    @staticmethod
    @abstractmethod
    def _encodingversion() -> str:
        pass

    def _write_attributes(self, obj: Union[h5py.Dataset, h5py.Group]):
        obj.attrs["encoding-type"] = self._encodingtype()
        obj.attrs["encoding-version"] = self._encodingversion()

        self._write_attributes_impl(obj)

    def _write_attributes_impl(self, obj: Union[h5py.Dataset, h5py.Group]):
        pass

    @property
    def backing(self) -> Union[h5py.Group, None, h5py.Dataset]:
        return self._backing

    def set_backing(
        self, parent: Optional[Union[h5py.Group, h5py.Dataset]] = None, key: Optional[str] = None
    ):
        if parent is not None:
            obj = self._writeable_object(parent, key) if key is not None else parent
            self._write_attributes(obj)
        else:
            obj = None
        self._set_backing(obj)
        if obj is None and self._backing is not None and self._backing.name == "/":
            warnings.warn("who is calling this?")
            # TODO: BUG: corrupted h5smu files after os._exit(0)
            # curerntly if we call for instance os._exit(0) after having modified a spatialmuon object, there can be
            # problems in saving the data to disk and the object can get corrupted (see the h5clear code in
            # spatialmudata.py). This could be because we need to close the object manually. Notice that in the
            # following we are closing the object only if the condition of the if are met, not in the general case.
            # This could be the source of the problem. Note also that if we close also obj, we need to open in again
            # before calling set_backing. That could be the right approach
            self._backing.close()
        # at this point self._backing can be either None, in that case we just set it to obj, or it can be referring
        # to a portion of an hdf5 file, like when we have (say) some RasterMasks from a Region and we copy them to
        # another Region object. In that case, the following line is what makes the new Region have a RasterMask
        # object that is pointing to the new portion of the hdf5 file
        self._backing = obj

    @abstractmethod
    def _set_backing(self, value: Optional[Union[h5py.Group, h5py.Dataset]] = None):
        pass

    @property
    def isbacked(self) -> bool:
        return self.backing is not None

    def write(self, parent: h5py.Group, key: Optional[str] = None):
        obj = self._writeable_object(parent, key) if key is not None else parent
        if self.isbacked:
            if self.backing.file != obj.file or self.backing.name != obj.name:
                obj.parent.copy(self.backing, os.path.basename(obj.name))
            else:
                print("to study the code")
        else:
            self._write_attributes(obj)
            self._write(obj)

    def _writeable_object(self, parent: h5py.Group, key: str) -> Union[h5py.Group, h5py.Dataset]:
        return parent.require_group(key) if key is not None else parent

    @abstractmethod
    def _write(self, obj: Union[h5py.Group, h5py.Dataset]):
        pass


class BackedDictProxy(UserDict):
    def __init__(
        self,
        parent: Optional[BackableObject] = None,
        key: Optional[str] = None,
        items: Optional[dict] = None,
        validatefun: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        if items is not None:
            items.update(kwargs)
        else:
            items = kwargs

        self._parent = parent if parent is not None else self
        self._key = key
        self.validatefun = validatefun
        self._grp = None

        for k, v in items.items():
            self.__setitem__(k, v)

    def _initgrp(self):
        if self._grp is None:
            if self._key is not None:
                self._grp = self._parent.backing.require_group(self._key)
            else:
                self._grp = self._parent.backing

    def __setitem__(self, key: str, value: BackableObject):
        if self.validatefun is not None:
            valid = self.validatefun(self._parent, key, value)
            if valid is not None:
                raise ValueError(valid)
        if self._parent.isbacked:
            self._initgrp()
            if key in self._grp and value.backing != self._grp[key] or key not in self._grp:
                value.set_backing(self._grp, key)
        else:
            print("to set a breakpoint and study the code")
        super().__setitem__(key, value)

    def __delitem__(self, key: str):
        super().__delitem__(key)
        if self._parent.isbacked:
            self._initgrp()
            del self._grp[key]
