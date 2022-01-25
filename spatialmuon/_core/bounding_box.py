import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from operator import xor
import copy


class BoundingBox:
    def __init__(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        z0: Optional[float] = None,
        z1: Optional[float] = None,
    ):
        assert not xor(z0 is None, z1 is None)
        if z0 is None:
            self.ndim = 2
        else:
            self.ndim = 3
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

    def validate(self):
        assert self.x0 < self.x1
        assert self.y0 < self.y1
        if self.ndim == 3:
            assert self.z0 < self.z1

    def __eq__(self, other):
        equal = True
        equal = equal and self.ndim == other.ndim
        equal = equal and np.isclose(self.x0, other.x0)
        equal = equal and np.isclose(self.x1, other.x1)
        equal = equal and np.isclose(self.y0, other.y0)
        equal = equal and np.isclose(self.y1, other.y1)
        if self.ndim == 3:
            equal = equal and np.isclose(self.z0, other.z0)
            equal = equal and np.isclose(self.z1, other.z1)
        return equal

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f'(x0: {self.x0}, x1: {self.x1}, y0: {self.y0}, y1: {self.y1})'

    def __repr__(self):
        return str(self)


class BoundingBoxable(ABC):
    def __int__(self):
        self.d = {}

    @property
    @abstractmethod
    def anchor(self):
        pass

    @property
    def bounding_box(self) -> BoundingBox:
        ndim = self.anchor.ndim
        assert ndim in [2, 3]
        if ndim == 3:
            raise NotImplementedError("3D case not yet implemented")
        bb = self.anchor.transform_bounding_box(self._untransformed_bounding_box)
        return bb

    @property
    @abstractmethod
    def _untransformed_bounding_box(self) -> BoundingBox:
        pass
