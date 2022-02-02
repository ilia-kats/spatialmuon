import numpy as np
from abc import ABC, abstractmethod
from spatialmuon.utils import angle_between


class BoundingBoxable(ABC):
    def __int__(self):
        self.d = {}

    @property
    @abstractmethod
    def anchor(self):
        pass

    @property
    def bounding_box(self) -> dict[str, float]:
        ndim = self.anchor.ndim
        assert ndim in [2, 3]
        if ndim == 3:
            raise NotImplementedError("3D case not yet implemented")
        origin = self.anchor.origin[...]
        vector = self.anchor.vector[...]
        ubb = self._untransformed_bounding_box
        self._validate_dict(ubb)

        # rotation
        # fmt: off
        rotation_matrix = np.array([
            [vector[0], -vector[1]],
            [vector[1], vector[0]]
        ])
        # fmt: on
        corners = np.array(
            [
                [ubb["x0"], ubb["y0"]],
                [ubb["x0"], ubb["y1"]],
                [ubb["x1"], ubb["y0"]],
                [ubb["x1"], ubb["y1"]],
            ]
        ).T
        rotated_corners = (rotation_matrix @ corners).T
        bb = {
            "x0": np.min(rotated_corners[:, 0]),
            "x1": np.max(rotated_corners[:, 0]),
            "y0": np.min(rotated_corners[:, 1]),
            "y1": np.max(rotated_corners[:, 1]),
        }
        self._validate_dict(bb)

        # translation
        bb["x0"] += origin[0]
        bb["x1"] += origin[0]
        bb["y0"] += origin[1]
        bb["y1"] += origin[1]
        self._validate_dict(bb)

        return bb

    @property
    @abstractmethod
    def _untransformed_bounding_box(self) -> dict[str, float]:
        pass

    def _validate_dict(self, d: dict[str, float]):
        assert all([type(dd) == float or type(dd) == np.float64 for dd in d.values()])
        ndim = self.anchor.ndim
        assert ndim in [2, 3]
        assert d["x0"] < d["x1"]
        assert d["y0"] < d["y1"]
        if ndim == 2:
            assert len(d.keys()) == 4
        elif ndim == 3:
            assert d["z0"] < d["z1"]
            assert len(d.keys()) == 6
