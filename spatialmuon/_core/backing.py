from abc import ABC, abstractmethod
from typing import Optional, Union

import h5py

class BackableObject(ABC):
    def __init__(self, backing: Optional[Union[h5py.Group, h5py.Dataset]]=None):
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
    def backing(self) -> Union[h5py.Group, None]:
        return self._backing

    @backing.setter
    def backing(self, value:Optional[h5py.Group]=None):
        if value is not None:
            self._write_attributes(value)
        self._set_backing(value)
        self._backing = value

    @abstractmethod
    def _set_backing(self, value: Optional[h5py.Group]=None):
        pass

    @property
    def isbacked(self) -> bool:
        return self.backing is not None

    @abstractmethod
    def write(self, parent:h5py.Group, key:str):
        pass
