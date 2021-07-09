import os
from abc import ABC, abstractmethod
from typing import Optional, Union

import h5py


class BackableObject(ABC):
    def __init__(self, backing: Optional[Union[h5py.Group, h5py.Dataset]] = None):
        super().__init__()
        self._backing = backing

    @staticmethod
    @abstractmethod
    def _encoding() -> str:
        pass

    @staticmethod
    @abstractmethod
    def _encodingversion() -> str:
        pass

    def _write_attributes(self, obj: Union[h5py.Dataset, h5py.Group]):
        obj.attrs["encoding"] = self._encoding()
        obj.attrs["encoding-version"] = self._encodingversion()

        self._write_attributes_impl(obj)

    @abstractmethod
    def _write_attributes_impl(self, obj: Union[h5py.Dataset, h5py.Group]):
        pass

    @property
    def backing(self) -> Union[h5py.Group, None, h5py.Dataset]:
        return self._backing

    def set_backing(self, parent: Optional[Union[h5py.Group, h5py.Dataset]] = None, key:Optional[str]=None):
        if parent is not None:
            obj = self._writeable_object(parent, key) if key is not None else parent
            self._write_attributes(obj)
        else:
            obj = None
        self._set_backing(obj)
        if obj is None and self._backing.name == "/":
            self._backing.close()
        self._backing = obj

    @abstractmethod
    def _set_backing(self, value: Optional[Union[h5py.Group, h5py.Dataset]] = None):
        pass

    @property
    def isbacked(self) -> bool:
        return self.backing is not None

    def write(self, parent: h5py.Group, key: str):
        obj = self._writeable_object(parent, key)
        self._write_attributes(obj)
        if self.isbacked:
            if self.backing.file != obj.file or self.backing.name != obj.name:
                obj.parent.copy(self.backing, os.path.basename(obj.name))
        else:
            self._write(obj)

    def _writeable_object(self, parent: h5py.Group, key: str) -> Union[h5py.Group, h5py.Dataset]:
        return parent.require_group(key) if key is not None else parent

    @abstractmethod
    def _write(self, obj: Union[h5py.Group, h5py.Dataset]):
        pass
