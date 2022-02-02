from __future__ import annotations
import os
from abc import ABC, abstractmethod
from collections import UserDict
from typing import Optional, Union, Callable, List, Dict
import copy
import warnings

import h5py


class BackableObject(ABC, UserDict):
    def __init__(
        self,
        backing: Optional[Union[h5py.Group, h5py.Dataset]] = None,
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

        self._key = key
        self.validatefun = validatefun

        for k, v in items.items():
            self.__setitem__(k, v)

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

    @abstractmethod
    def _write_attributes_impl(self, obj: Union[h5py.Dataset, h5py.Group]):
        pass

    @property
    def backing(self) -> Union[h5py.Group, None, h5py.Dataset]:
        return self._backing

    @property
    def is_backed(self) -> bool:
        return self.backing is not None

    def set_backing(
        self, parent: Optional[Union[h5py.Group, h5py.Dataset]] = None, key: Optional[str] = None
    ):
        # if parent is not None:
        #     obj = parent.require_group(key) if key is not None else parent
        #     self._write_attributes(obj)
        # else:
        #     obj = None
        # self._set_backing(obj)
        if parent is not None:
            obj = self._write(parent, key=key)
            for k, v in self.items():
                child_obj = v.set_backing(obj._backing, k)
                if id(child_obj) != id(self[k]):
                    super().__setitem__(k, child_obj)
            if obj is None and self._backing is not None and self._backing.name == "/":
                assert False, "Branch never tested, set a breakpoint and study"
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
            # self._backing = obj
            return obj
        else:
            assert False, "branch never tested"
            assert self.is_backed
        # TODO: check propagation to childs is done

    def _write(self, parent: h5py.Group, key: str):
        if self.is_backed:
            # if we are copying from a different file or from a different location in the file
            if self.backing.file != parent.file or self.backing.name != os.path.join(
                parent.name, key
            ):
                # if the key is already present in the parent is not because the object was already populated but
                # because the shallow copy was called before
                # example: we have a backed mod with a fov inside, and we copy the mod into a new SpatialMuData backed
                # object.
                # Then, when copy the mod into the SpatialMuData object, the shallow copy will create already the
                # group for the fov, but it will be empty, since it will be populated from another call of _write(),
                # invoked from set_backing() in the "for k, v in self.items()" loop
                if key in parent:
                    assert len(parent[key]) == 0
                    # actually attributes are copied from subgroups, so this assertion would be false, but it is fine
                    # since we are gonna rewrite them in the parent.copy(...) call below
                    # assert len(parent[key].attrs) == 0
                    # let's clean the target object otherwise the copy can't be performed
                    del parent[key]
                parent.copy(self.backing, key, shallow=True)
                new_backable = copy.copy(self)
                new_backable._backing = parent[key]
                return new_backable
                # return parent[key]
            else:
                assert False, "to study the code 1"
        else:
            obj = parent.require_group(key) if key is not None else parent
            self._write_attributes(obj)
            self._write_impl(obj)
            self._backing = obj
            return self

    @abstractmethod
    def _write_impl(self, obj: Union[h5py.Group, h5py.Dataset]):
        pass

    @property
    def _grp(self):
        if self._key is not None:
            return self.backing.require_group(self._key)
        else:
            return self.backing

    def __setitem__(self, key: str, value: BackableObject):
        value_in_memory = value
        if self.validatefun is not None:
            valid = self.validatefun(self, key, value)
            if valid is not None:
                raise ValueError(valid)
        if self.is_backed:
            if key not in self._grp:
                value_in_memory = value.set_backing(self._grp, key)
        else:
            # only the python dict (the superclass) is updated, not the file; the file gets updated in the case
            # in which this object (or a parent object invoking the backing downstream) is assigned to a
            # parent object that is backed
            pass
        super().__setitem__(key, value_in_memory)

    def __delitem__(self, key: str):
        super().__delitem__(key)
        if self.is_backed:
            del self._grp[key]