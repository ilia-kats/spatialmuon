from __future__ import annotations

import numpy as np
from typing import Optional, Union
import warnings
from spatialmuon._core.backing import BackableObject
import h5py
from spatialmuon.utils import _read_hdf5_attribute


class Anchor(BackableObject):
    """This class is used to orient spatial data in a globale coordinate space.

    Spatial modalities can be aligned and oriented to eachother in a m:n fashion
    depending on the experimental data. This is implemented in spatialmuon as an
    (origin/vector) per modality, referred to as 'Anchor'. When aligning two
    modalities, the translation, rotation and scaling can be calculated on the
    fly based on two anchors. The given components refer to a global coordinate
    system so that no individual references for each alignment pair have to be
    saved.
    """

    def __init__(
        self,
        ndim: Optional[int] = None,
        origin: Optional[np.ndarray] = None,
        vector: Optional[np.ndarray] = None,
        backing: Optional[Union[h5py.Group, h5py.Dataset]] = None,
    ):
        super().__init__(backing)
        if backing is not None:
            if ndim is not None or origin is not None or vector is not None:
                raise ValueError("trying to set attributes for a non-empty backing store")
            else:
                self._origin = backing["origin"]
                self._vector = backing["vector"]
                self._ndim = _read_hdf5_attribute(backing.attrs, "ndim")
        else:
            if ndim is None and origin is None and vector is None:
                raise ValueError("at least one parameter should be specified")

            if ndim is not None:
                self._ndim = ndim
            if origin is not None:
                self._ndim = len(origin)
                if ndim is not None:
                    assert len(origin) == ndim
            if vector is not None:
                self._ndim = len(vector)
                if ndim is not None:
                    assert len(vector) == ndim
                if origin is not None:
                    assert len(origin) == len(vector)


            if origin is None:
                self._origin = np.array([0] * self.ndim)
            else:
                self._origin = origin

            if vector is None:
                self._vector = np.array([1] + ([0] * (self.ndim - 1)))
            else:
                self._vector = vector

    def _set_backing(self, value: Optional[Union[h5py.Group, h5py.Dataset]] = None):
        self._write(value)

    def _write(self, obj: Union[h5py.Group, h5py.Dataset]):
        obj.create_dataset("origin", data=self.origin)
        obj.create_dataset("vector", data=self.vector)

    def _write_attributes_impl(self, obj: Union[h5py.Dataset, h5py.Group]):
        obj.attrs["ndim"] = self.ndim

    @staticmethod
    def _encodingtype() -> str:
        return "anchor"

    @staticmethod
    def _encodingversion() -> str:
        return "0.1.0"

    def __str__(self):
        return "{}\n├─ndim: {}\n├─origin: {}\n└─vector: {}".format(
            self.__class__.__name__, self.ndim, self.origin, self.vector
        )

    @property
    def ndim(self) -> int:
        """Number of dimensions of the Anchor point.

        An Anchor point contains two np.ndarrays. These must have the same
        number of components as defined through 'ndim'. When the Anchor is
        initialised empty, 'ndim' is used to initialise 'origin' and 'vector'
        with the following default values:

        origin = np.array([0] * ndim)
        vector = np.array([1] + ([0] * (ndim - 1)))
        """
        assert self._ndim is not None
        return self._ndim

    @property
    def origin(self) -> np.ndarray:
        """Origin of the Anchor in the global coordinate space.

        Spatial modalities are embedded in a global coordinate space using a
        (origin/vector) pair. When the modalities are then aligned, the global
        system is used so that no alignment-pairs have to be stored. This
        property hold a np.ndarray containing the 'origin' point with n
        dimensions. Currently, spatialmuon supports the 2D and 3D case in its
        helper functions.

        np.ndarray[0] | np.ndarray[1] | np.ndarray[2] |       ...
        ----------------------------------------------------------------
              X       |       Y       |       Z       | Not implemented

        """
        assert self._origin is not None
        return self._origin

    @origin.setter
    def origin(self, new_origin: np.ndarray):
        """Updates the np.ndarray holding the 'origin'."""
        print(2)
        if not isinstance(new_origin, np.ndarray):
            raise TypeError("Please specify a np.ndarray for 'origin'.")
        if len(new_origin) == 0:
            raise ValueError("Won't assign empty np.ndarray to 'origin'.")
        if len(new_origin) > self.ndim:
            w = "'origin' too long, using only the first {}.".format(self.ndim)
            warnings.warn(w)
        if len(new_origin) < self.ndim:
            raise ValueError("Length of 'new_origin' must be same as current 'ndim'.")
        self.origin = new_origin[: self.ndim]

    @property
    def vector(self) -> np.ndarray:
        """Vector of the Anchor in the global coordinate space.

        Spatial modalities are embedded in a global coordinate space using a
        (origin/vector) pair. When the modalities are then aligned, the global
        system is used so that no alignment-pairs have to be stored. This
        property hold a np.ndarray containing the 'vector' point with n
        dimensions. Currently, spatialmuon supports the 2D and 3D case in its
        helper functions.

        np.ndarray[0] | np.ndarray[1] | np.ndarray[2] |       ...
        ----------------------------------------------------------------
              X       |       Y       |       Z       | Not implemented

        """
        assert self._vector is not None
        return self._vector

    @property
    def scale_factor(self) -> float:
        return np.linalg.norm(self.vector)

    @property
    def normalized_vector(self) -> np.ndarray:
        return self.vector / self.scale_factor

    @vector.setter
    def vector(self, new_vector: np.ndarray):
        """Updates the np.ndarray holding the 'vector'."""
        print(2)
        if not isinstance(new_vector, np.ndarray):
            raise TypeError("Please specify a np.ndarray for 'vector'.")
        if len(new_vector) == 0:
            raise ValueError("Won't assign empty np.ndarray to 'vector'.")
        if len(new_vector) > self.ndim:
            w = "New 'vector' too long, using only the first {}.".format(self.ndim)
            warnings.warn(w)
        if len(new_vector) < self.ndim:
            raise ValueError("Length of 'new_vector' must be same as current 'ndim'.")
        self.vector = new_vector[: self.ndim]

    # flake8: noqa: C901
    def move_origin(self, axis: str = "all", distance: Union[int, float] = 0):
        """Additively translates components of the Anchor's 'origin'.

        Instead of defining a new 'origin' in the Anchor, the components can
        also be iteratively translated along the axes.

        Parameters:
          axis (str): Axis along which to shift (valid: all, x, y, z)
          distance (float): Distance to shift along axis

        """

        if not isinstance(axis, str):
            raise TypeError("Please specify a str for 'axis'.")

        if not (isinstance(distance, float) or isinstance(distance, int)):
            raise TypeError("Please specify an int or float for 'distance'.")

        if self.ndim == 1:

            if axis.lower() == "x" or axis.lower() == "all":
                self.origin[0] += distance
            else:
                raise ValueError("Invalid choice for 'axis', use (all/x).")

        if self.ndim == 2:

            if axis.lower() == "x":
                self.origin[0] += distance
            elif axis.lower() == "y":
                self.origin[1] += distance
            elif axis.lower() == "all":
                self.origin[0] += distance
                self.origin[1] += distance
            else:
                raise ValueError("Invalid choice for 'axis', use (all/x/y).")

        if self.ndim == 3:

            if axis.lower() == "x":
                self.origin[0] += distance
            elif axis.lower() == "y":
                self.origin[1] += distance
            elif axis.lower() == "z":
                self.origin[2] += distance
            elif axis.lower() == "all":
                self.origin[0] += distance
                self.origin[1] += distance
                self.origin[2] += distance
            else:
                raise ValueError("Invalid choice for 'axis', use (all/x/y/z).")

    def rotate_vector(self, angle: Union[int, float] = 0):
        """Rotates components of the Anchor's 'vector'.

        Instead of setting a new 'vector' in the Anchor, the existing 'vector'
        can also be rotated.

        WARNING: The rotation is calculated in floating-point format.

        Parameters:
          angle (int): Angle in degrees to rotate the vector, can be positive or
            negative. Positive angles will be used as "counterclockwise", while
            negative will count as "clockwise".

        """

        # todo(ttreis): Implement 3D rotation once we have a usecase

        if not (isinstance(angle, int) or isinstance(angle, float)):
            raise TypeError("Please specify an int or float for 'angle'.")

        if self.ndim == 1:
            raise ValueError("Rotating in 1D space is useless.")
        elif self.ndim == 2:
            current_vector = self.vector
            theta = np.deg2rad(angle)
            theta_cos = np.cos(theta)
            theta_sin = np.sin(theta)
            x = current_vector[0] * theta_cos - current_vector[1] * theta_sin
            y = current_vector[0] * theta_sin + current_vector[1] * theta_cos
            self._vector = np.array([x, y])
        elif self.ndim >= 3:
            raise NotImplementedError("Not yet implemented.")

    def scale_vector(self, factor: float = 1.0):
        """Scales the Anchor's 'vector'.

        This function can scale the anchor my multiplying the current 'vector's
        length with 'factor'. This can be used to adjust the size of modalities.

        WARNING: The rotation is calculated in floating-point format.

        Parameters:
          factor (float): A skalar that will be multiplied with the current
            'vector's length.
        """

        # todo(ttreis): Implement 3D rotation once we have a usecase

        if not isinstance(factor, float):
            try:
                factor = float(factor)
            except ValueError:
                raise ValueError("Please specify a float for 'factor'.")

        self._vector = np.array([i * factor for i in self.vector])

    def transform_coordinates(self, coords: np.ndarray) -> np.ndarray:
        old_shape = coords.shape
        if len(coords.shape) in [0, 1] and len(coords) == 2:
            coords = coords.reshape((1, 2))
        assert len(coords.shape) == 2
        if not coords.shape[1] == 2:
            raise NotImplementedError('only the 2D case is currently implemented')
        # rotation
        # fmt: off
        cos, sin = self.normalized_vector
        rotation_matrix = np.array([
            [cos, -sin],
            [sin, cos]
        ])
        # fmt: on
        rotated = (rotation_matrix @ coords.T).T
        scaled = rotated / self.scale_factor
        translated = scaled + self.origin
        translated = translated.reshape(old_shape)
        return translated