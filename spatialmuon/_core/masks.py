from __future__ import annotations

import abc
from collections.abc import MutableMapping
from abc import abstractmethod
from typing import Optional, Union, NewType

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.axes
import numpy as np
import h5py
from shapely.geometry import Polygon
from trimesh import Trimesh
from skimage.measure import find_contours, marching_cubes
from anndata._io.utils import read_attribute, write_attribute
import pandas as pd
import skimage.measure

from spatialmuon._core.fieldofview import FieldOfView
from spatialmuon._core.backing import BackableObject
from spatialmuon.utils import _read_hdf5_attribute, UnknownEncodingException

# either a color or a list of colors
ColorsType = Optional[
    Union[
        str,
        list[str],
        np.ndarray,
        list[np.ndarray],
        list[int],
        list[float],
        list[list[int]],
        list[list[float]],
    ]
]

class Masks(BackableObject):
    def __new__(cls, *, backing: Optional[h5py.Group] = None, **kwargs):
        if backing is not None:
            masktype = _read_hdf5_attribute(backing.attrs, "encoding-type")
            if masktype == "masks-polygon":
                return super(cls, PolygonMasks).__new__(PolygonMasks)
            elif masktype == "masks-raster":
                return super(cls, RasterMasks).__new__(RasterMasks)
            elif masktype == "masks-mesh":
                return super(cls, MeshMasks).__new__(MeshMasks)
            else:
                raise UnknownEncodingException(masktype)
        else:
            return super().__new__(cls)

    def __init__(
        self,
        obs: Optional[pd.DataFrame] = None,
        backing: Optional[Union[h5py.Group, h5py.Dataset]] = None,
    ):
        super().__init__(backing)
        self._parentdataset = None
        if backing is not None:
            self._obs = read_attribute(backing["obs"])
        else:
            if obs is not None:
                self._obs = obs
            else:
                self._obs = pd.DataFrame()

    @property
    def parentdataset(self):
        return self._parentdataset

    @parentdataset.setter
    def parentdataset(self, dset: FieldOfView):
        self._parentdataset = dset

    @property
    @abstractmethod
    def ndim(self):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __len__(self):
        pass

    # @abstractmethod
    def update_obs_from_masks(self):
        pass

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @property
    def n_obs(self) -> int:
        return self._obs.shape[0]

    def _write_data(self, grp):
        if "geometry" in self._obs.columns:
            o = (self._obs.drop(self._obs.geometry.name, axis=1),)
        else:
            o = self._obs
        write_attribute(
            grp,
            "obs",
            o,
            dataset_kwargs={"compression": "gzip", "compression_opts": 9},
        )

    def _set_backing(self, obj=None):
        self._write_data(obj)

    def __repr__(self, mask_type="masks"):
        repr_str = f"│   ├── {self.ndim}D {mask_type} with {self.n_obs} obs: {', '.join(self.obs)}"
        repr_str = "│   └──".join(repr_str.rsplit("│   ├──", 1))
        return repr_str

    @abstractmethod
    def plot(self):
        pass


class ShapeMasks(Masks, MutableMapping):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        masks_dict: Optional[dict[str, tuple[tuple[float], float]]] = None,
        obs: Optional[pd.DataFrame] = None,
    ):
        super().__init__(obs, backing)

        self._data = {}
        self._ndim = None

        if masks_dict is not None:
            if self.isbacked and len(self.backing) > 0:
                raise ValueError("trying to set masks on a non-empty backing store")
            for key, mask in masks_dict.items():
                ndim = len(mask[0])
                if self.ndim is None:
                    self._ndim = ndim
                if ndim != self.ndim:
                    raise ValueError("all masks must have the same dimensionality")

            if self.isbacked:
                self[key] = mask
            else:
                self._data[key] = mask

    @property
    def ndim(self):
        return self._ndim

    def __getitem__(self, key) -> Polygon:
        if self.isbacked:
            return (self.backing[key]["center"], self.backing[key]["radius"])
        else:
            return self._data[key]

    def __setitem__(self, key: str, value: tuple[tuple[float], float]):
        if self.ndim is not None and self.ndim != len(value[0]):
            raise ValueError(f"value must have dimensionality {self.ndim}, but has {len(value[0])}")
        if self.isbacked:
            grp = self.backing.create_group(key)
            grp.create_dataset("center", data=value[0])
            grp.create_dataset("radius", data=np.array([value[1]]))
        else:
            self._data[key] = value

    def __delitem__(self, key: str):
        if self.isbacked:
            del self.backing[key]
        else:
            del self._data[key]

    def __len__(self):
        if self.isbacked:
            return len(self.backing)
        else:
            return len(self._data)

    def __contains__(self, item):
        if self.isbacked:
            return item in self.backing
        else:
            return item in self._data

    def __iter__(self):
        if self.isbacked:
            return iter(self.backing)
        else:
            return iter(self._data)

    def items(self):
        raise NotImplementedError()

    def values(self):
        raise NotImplementedError()

    @staticmethod
    def _encodingtype():
        return "masks-shape"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, value: h5py.Group):
        super()._set_backing(value)
        if value is None and self.backed:
            for k, v in self.backing.items():
                self._data[k] = (self.backing["center"][:], self.backing["radius"][:])
        elif value is not None:
            self._write(value)
            self._data.clear()

    def _write(self, obj: h5py.Group):
        for k, v in self._data.items():
            grp = obj.create_group(k)
            grp.create_dataset("center", v[0])
            grp.create_dataset("radius", np.array([v[1]]))


class PolygonMasks(Masks, MutableMapping):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        masks: Optional[dict[str, Union[np.ndarray, Polygon]]] = None,
    ):
        super().__init__(backing)

        self._data = {}
        self._ndim = None
        if masks is not None:
            if self.isbacked and len(self.backing) > 0:
                raise ValueError("trying to set masks on a non-empty backing store")
            for key, mask in masks:
                if isinstance(mask, Polygon):
                    ndim = 3 if mask.has_z else 2
                else:
                    ndim = mask.ndim
                if self.ndim is None:
                    self.ndim = ndim
                if ndim != self.ndim:
                    raise ValueError("all masks must have the same dimensionality")

                self[key] = mask

    @property
    def ndim(self):
        return self._ndim

    def __getitem__(self, key) -> Polygon:
        if self.isbacked:
            return Polygon(self.backing[key][:])
        else:
            return self._data[key]

    def __setitem__(self, key: str, value: Union[np.ndarray, Polygon]):
        if self.isbacked:
            if isinstance(value, Polygon):
                value = np.asarray(value.exterior.coords)
            else:
                value = np.asarray(value)
            if self.ndim is not None and self.ndim != value.ndim:
                raise ValueError(
                    f"value must have dimensionality {self.ndim}, but has {value.ndim}"
                )
            self.backing.create_dataset(key, data=value, compression="gzip", compression_opts=9)
        else:
            if not isinstance(value, Polygon):
                ndim = value.ndim
                value = Polygon(value)
            else:
                ndim = 3 if value.has_z else 2
            if self.ndim is not None and self.ndim != ndim:
                raise ValueError(f"value must have dimensionality {self.ndim}, but has {ndim}")
            self._data[key] = value

    def __delitem__(self, key: str):
        if self.isbacked:
            del self.backing[key]
        else:
            del self._data[key]

    def __len__(self):
        if self.isbacked:
            return len(self.backing)
        else:
            return len(self._data)

    def __contains__(self, item):
        if self.isbacked:
            return item in self.backing
        else:
            return item in self._data

    def __iter__(self):
        if self.isbacked:
            return iter(self.backing)
        else:
            return iter(self._data)

    def items(self):
        raise NotImplementedError()

    def values(self):
        raise NotImplementedError()

    @staticmethod
    def _encodingtype():
        return "masks-polygon"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, value: h5py.Group):
        super()._set_backing(value)
        if value is None and self.backed:
            for k, v in self.backing.items():
                self._data[k] = Polygon(v[:])
        elif value is not None:
            self._write(value)
            self._data.clear()

    def _write(self, obj: h5py.Group):
        for k, v in self._data.items():
            obj.create_dataset(k, data=v.exterior.coords, compression="gzip", compression_opts=9)


class MeshMasks(Masks, MutableMapping):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        masks: Optional[dict[str, Union[Trimesh, tuple[np.ndarray, np.ndarray]]]] = None,
    ):
        super().__init__(backing)
        self._data = {}
        if masks is not None:
            if self.isbacked and len(self.backing) > 0:
                raise ValueError("trying to set masks on a non-empty backing store")
            self.update(masks)

    @property
    def ndim(self):
        return 3

    def __getitem__(self, key) -> Trimesh:
        if self.isbacked:
            return Trimesh(
                vertices=self.backing[key]["vertices"][()], faces=self.backing[key]["faces"][()]
            ).fix_normals()
        else:
            return self._data[key]

    def __setitem__(self, key: str, value: Union[Trimesh, tuple[np.ndarray, np.ndarray]]):
        if self.isbacked:
            if isinstance(value, Trimesh):
                vertices = value.vertices
                faces = value.faces
            else:
                vertices, faces = value
                if vertices.shape[1] != 3:
                    raise ValueError(
                        f"masks must be 3-dimensional, but masks {key} has dimensionality {vertices.shape[1]}"
                    )
                if faces.shape[1] != 3 and faces.shape[1] != 4:
                    raise ValueError(
                        f"faces must reference 3 or 4 vertices, but faces for masks {key} reference {faces.shape[1]} vertices"
                    )
                maxface = faces.max()
                if maxface > vertices.shape[0] - 1:
                    raise ValueError(
                        f"masks {key} has {vertices.shape[0]} vertices, but faces reference up to {maxface} vertices"
                    )

            self.backing.create_dataset(
                f"{key}/vertices", data=vertices, compression="gzip", compression_opts=9
            )
            self.backing.create_dataset(
                f"{key}/faces", data=faces, compression="gzip", compression_opts=9
            )

    def __delitem__(self, key: str):
        if self.isbacked:
            del self.backing[key]
        else:
            del self._data[key]

    def __len__(self):
        if self.isbacked:
            return len(self.backing)
        else:
            return len(self._data)

    def __contains__(self, item):
        if self.isbacked:
            return item in self._backing
        else:
            return item in self._data

    def __iter__(self):
        if self.isbacked:
            return iter(self.backing)
        else:
            return iter(self._data)

    def items(self):
        raise NotImplementedError()

    def values(self):
        raise NotImplementedError()

    @staticmethod
    def _encodingtype():
        return "masks-mesh"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, value: h5py.Group):
        super()._set_backing(value)
        if value is None and self.backed:
            for k in self.keys():
                self._data[k] = self[k]
        elif value is not None:
            self._write(value)
            self._data.clear()

    def _write(self, obj: h5py.Group):
        for k, v in self._data.items():
            self.backing.create_dataset(
                f"{k}/vertices", data=v.vertices, compression="gzip", compression_opts=9
            )
            self.backing.create_dataset(
                f"{k}/faces", data=v.faces, compression="gzip", compression_opts=9
            )


class RasterMasks(Masks):
    def __init__(
        self,
        backing: Optional[h5py.Dataset] = None,
        mask: Optional[np.ndarray] = None,
        shape: Optional[Union[tuple[int, int], tuple[int, int, int]]] = None,
        dtype: Optional[type] = None,
    ):
        super().__init__(backing=backing)
        self._mask = None

        if mask is not None:
            if self.isbacked and self.backing.size > 0:
                raise ValueError("attempting to set masks on a non-empty backing store")
            if mask.ndim < 2 or mask.ndim > 3:
                raise ValueError("masks must have 2 or 3 dimensions")
            if not np.issubdtype(mask.dtype, np.unsignedinteger):
                raise ValueError("masks must have an unsigned integer dtype")
            self._mask = mask
        elif not self.isbacked:
            if shape is None or dtype is None:
                raise ValueError("if masks is None shape and dtype must be given")
            if len(shape) < 2 or len(shape) > 3:
                raise ValueError("shape must have 2 or 3 dimensions")
            if not np.issubdtype(dtype, np.unsignedinteger):
                raise ValueError("dtype must be an unsigned integer type")
            self._mask = np.zeros(shape=shape, dtype=dtype)

    @property
    def ndim(self):
        if self.isbacked:
            return self.backing["imagemask"].ndim
        else:
            return self._mask.ndim

    @property
    def shape(self):
        if self.isbacked:
            return self.backing.shape
        else:
            return self._mask.shape

    @property
    def dtype(self):
        if self.isbacked:
            return self.backing.dtype
        else:
            return self._mask.dtype

    @property
    def data(self) -> Union[np.ndarray, h5py.Dataset]:
        if self.isbacked:
            return self.backing["imagemask"][...]
        else:
            return self._mask

    def __len__(self):
        return np.max(self.data)

    # flake8: noqa: C901
    def __getitem__(self, key):
        if not np.issubdtype(type(key), np.integer):
            raise TypeError("key must be an integer")
        if key >= 0:
            coords = np.where(self.data[()] == key + 1)
            if any(c.size == 0 for c in coords):
                raise KeyError(key)
            boundaries = []
            for i, c in enumerate(coords):
                min, max = np.min(c), np.max(c) + 1
                if min > 0:
                    min -= 1
                if max <= self.data.shape[i]:
                    max += 1
                boundaries.append(slice(min, max))
            cmask = self.data[tuple(boundaries)]
            cmask[cmask != key + 1] = 0
            if self.ndim == 2:
                contour = find_contours(cmask, fully_connected="high")
                for c in contour:
                    for d in range(self.ndim):
                        c[:, d] += boundaries[d].start
                return (
                    Polygon(contour[0][:, ::-1])
                    if len(contour) == 1
                    else [Polygon(c[:, ::-1]) for c in contour]
                )
                # TODO: scale by px_size and px_distance from parentdataset
            else:
                vertices, faces, normals, _ = marching_cubes(cmask, allow_degenerate=False)
                for d in range(self.ndim):
                    vertices[:, d] += boundaries[d].start
                    faces[:, d] += boundaries[d].start
                return Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
                # TODO: scale by px_size and px_distance from parentdataset

    @staticmethod
    def _encodingtype():
        return "masks-raster"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, value: h5py.Group):
        super()._set_backing(value)
        if value is None and self.backed:
            self._mask = self.backing["imagemask"]
        elif value is not None:
            self._write(value)
            self._mask = None

    def _write(self, obj: h5py.Group):
        obj.create_dataset("imagemask", data=self._mask, compression="gzip", compression_opts=9)

    def __repr__(self):
        return super().__repr__(mask_type="raster masks")

    def update_obs_from_masks(self):
        # if the dataframe is not empty
        if self._obs is not None and len(self._obs.columns) != 0:
            raise ValueError(
                "replacing the old obs is only performed when obs is an empty DataFrame or it is None"
            )
        if self._mask is None:
            raise ValueError("no mask data has been specified")
        unique_masks = np.unique(self._mask)
        mask_df = pd.DataFrame(data=dict(original_labels=unique_masks))
        self._obs = mask_df

    def plot(
        self,
        fill_colors: ColorsType = "random",
        outline_colors: ColorsType = None,
        background_color: Optional[Union[str, np.array]] = "black",
        ax: matplotlib.axes.Axes=None
    ):
        n = len(self._obs)

        def normalize_color(x):
            if type(x) == tuple:
                x = np.array(x)
            assert len(x.shape) in [0, 1]
            x = x.flatten()
            assert len(x) in [3, 4]
            if len(x) == 3:
                x = np.array(x.tolist() + [1.0])
            x = x.reshape(1, -1)
            return x

        def get_color_array_rgba(color):
            if color is None:
                return np.zeros((n, 4))
            elif type(color) == str and color == "random":
                a = np.random.rand(n, 3)
                b = np.ones(len(a)).reshape(-1, 1)
                c = np.concatenate((a, b), axis=1)
                return c
            elif type(color) == str:
                a = matplotlib.colors.to_rgba(color)
                b = [a] * n
                c = np.array(b)
                return c
            elif type(color) == list or type(color) == np.ndarray:
                try:
                    a = matplotlib.colors.to_rgba(color)
                    b = [normalize_color(a)] * n
                    c = np.concatenate(b, axis=0)
                    return c
                except ValueError:
                    b = []
                    for c in color:
                        d = matplotlib.colors.to_rgba(c)
                        b.append(normalize_color(d))
                    e = np.concatenate(b, axis=0)
                    if len(e) != n:
                        raise ValueError("number of colors must match the number of elements in obs")
                    return e
            else:
                raise ValueError(f"invalid way of specifying the color: {color}")

        fill_color_array = get_color_array_rgba(fill_colors)
        outline_colors_array = get_color_array_rgba(outline_colors)
        if background_color is not None:
            background_color = normalize_color(matplotlib.colors.to_rgba(background_color))
            fill_color_array[0] = background_color
            outline_colors_array[0] = background_color

        labels = self.obs.to_dict()["original_labels"]
        contiguous_labels = list(labels.keys())
        assert max(contiguous_labels) + 1 == len(contiguous_labels)
        original_labels = list(labels.values())
        x = self.data
        if original_labels != contiguous_labels:
            # probably not the most efficient, but probably also fine
            lut = np.zeros(max(original_labels) + 1, dtype=np.int)
            for i, o in enumerate(original_labels):
                lut[o] = i
            x = lut[self.data]
        if ax is None:
            plt.figure()
            axs = plt.gca()
        else:
            axs = ax
        axs.imshow(fill_color_array[x])

        if outline_colors is not None:
            for i, c in enumerate(contiguous_labels):
                outline_color = outline_colors_array[i]
                boolean_mask = x == c
                contours = skimage.measure.find_contours(boolean_mask, 0.7)
                for contour in contours:
                    axs.plot(contour[:, 1], contour[:, 0], linewidth=2, color=outline_color)
        if ax is None:
            plt.show()


if __name__ == "__main__":
    x = np.array(range(100), dtype=np.uint).reshape(10, 10)

    rm = RasterMasks(backing=None, mask=x)
    rm.update_obs_from_masks()
