from __future__ import annotations

import numpy as np
import pandas as pd


class Anchor:
    """This class is used to orient spatial data in a globale coordinate space.

    Spatial modalities can be aligned and oriented to eachother in a m:n fashion
    depending on the experimental data. This is implemented in spatialmuon as an
    (origin/vector) per modality, referred to as 'Anchor'. When aligning two
    modalities, the translation, rotation and scaling can be calculated on the
    fly based on two anchors. The given components refer to a global coordinate
    system so that no individual references for each alignment pair have to be
    saved.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(
        self, 
        ndim: int, 
        origin: Optional[np.ndarray] = None, 
        vector: Optional[np.ndarray] = None
    ):
        if origin is None:
            self._origin = np.array([0] * ndim)
        else:
            self.origin = origin

        if vector is None:
            self._vector = np.array([1] + ([0] * (ndim - 1)))
        else:
            self.vector = vector

        self._ndim = ndim

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
        if self._ndim is None:
            return self.ndim
        else:
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
        if self._origin is None:
            return self.origin
        else:
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
        if self._vector is None:
            return self.vector
        else:
            return self._vector

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

    def move_origin(self, axis: str = "all", distance: int = 0):
        """Additively translates components of the Anchor's 'origin'.

        Instead of defining a new 'origin' in the Anchor, the components can
        also be iteratively translated along the axes.

        Parameters:
          axis (str): Axis along which to shift (valid: all, x, y, z)
          distance (int): Distance to shift along axis

        """

        if not isinstance(axis, str):
            raise TypeError("Please specify a str for 'axis'.")

        if not isinstance(distance, int):
            raise TypeError("Please specify an int for 'distance'.")

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

    def rotate_vector(self, angle: int = 0):
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

        if not isinstance(angle, int):
            raise TypeError("Please specify an int for 'angle'.")

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
