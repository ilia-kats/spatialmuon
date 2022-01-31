from __future__ import annotations

import abc
import warnings
from collections.abc import MutableMapping
from abc import abstractmethod
from typing import Optional, Union, Literal
from scipy.ndimage import center_of_mass

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.axes
import matplotlib.cm
import matplotlib.patches
import matplotlib.collections
import numpy as np
import h5py
import copy
import math
from shapely.geometry import Polygon
from trimesh import Trimesh
from skimage.measure import find_contours, marching_cubes
from anndata._io.utils import read_attribute, write_attribute
import pandas as pd
import skimage.measure
import vigra
from enum import Enum, auto

from spatialmuon._core.fieldofview import FieldOfView
from spatialmuon._core.backing import BackableObject
from spatialmuon.utils import _read_hdf5_attribute, UnknownEncodingException, _get_hdf5_attribute
from spatialmuon._core.bounding_box import BoundingBoxable, BoundingBox

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


class SpotShape(Enum):
    circle = auto()
    rectangle = auto()

    def __str__(self):
        return str(self.name)


class Masks(BackableObject, BoundingBoxable):
    def __new__(cls, *, backing: Optional[h5py.Group] = None, **kwargs):
        if backing is not None:
            masktype = _read_hdf5_attribute(backing.attrs, "encoding-type")
            if masktype == "masks-polygon":
                return super(cls, PolygonMasks).__new__(PolygonMasks)
            elif masktype == "masks-raster":
                return super(cls, RasterMasks).__new__(RasterMasks)
            elif masktype == "masks-mesh":
                return super(cls, MeshMasks).__new__(MeshMasks)
            elif masktype == "masks-shape":
                return super(cls, ShapeMasks).__new__(ShapeMasks)
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
    def anchor(self):
        return self._parentdataset.anchor

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

    def _set_backing(self, grp: Optional[h5py.Group] = None):
        self._write_data(grp)

    def __repr__(self, mask_type="masks"):
        repr_str = f"│   ├── {self.ndim}D {mask_type} with {self.n_obs} obs: {', '.join(self.obs)}"
        repr_str = "│   └──".join(repr_str.rsplit("│   ├──", 1))
        return repr_str

    @abstractmethod
    def accumulate_features(self, x: Union["Raster", "Regions"]):
        pass

    @abstractmethod
    def extract_tiles(self, raster: "Raster", tile_dim: int):
        pass

    @staticmethod
    def normalize_color(x):
        if type(x) == tuple:
            x = np.array(x)
        assert len(x.shape) in [0, 1]
        x = x.flatten()
        assert len(x) in [3, 4]
        if len(x) == 3:
            x = np.array(x.tolist() + [1.0])
        # else:
        #     x[3] = 1.0
        x = x.reshape(1, -1)
        return x

    @staticmethod
    def apply_alpha(x, alpha):
        assert len(x.shape) == 2
        assert x.shape[1] == 4
        x[:, 3] *= alpha

    @abstractmethod
    def _plot(
        self,
        fill_color_array: np.ndarray,
        outline_colors_array: Optional[np.ndarray],
        ax: matplotlib.axes.Axes,
    ):
        pass

    def plot(
        self,
        fill_colors: ColorsType = "random",
        outline_colors: ColorsType = None,
        background_color: Optional[Union[str, np.array]] = (0.0, 0.0, 0.0, 0.0),
        ax: matplotlib.axes.Axes = None,
        alpha: float = 1.0,
        show_title: bool = True,
        show_legend: bool = True,
    ):
        # adding the background
        n = len(self._obs) + 1
        plotting_a_category = False
        title = None
        _legend = None

        # return a tensor of length n, then the first element (the background color), will be replaced with
        # background_color
        def get_color_array_rgba(color):
            if color is None:
                return np.zeros((n, 4))
            elif type(color) == str and color in self.obs.columns:
                nonlocal plotting_a_category
                nonlocal title
                nonlocal _legend
                plotting_a_category = True
                title = color
                cmap = matplotlib.cm.get_cmap("tab10")
                categories = self.obs[color].cat.categories.values.tolist()
                cycled_colors = list(cmap.colors) * (len(categories) // len(cmap.colors) + 1)
                d = dict(zip(categories, cycled_colors))
                levels = self.obs[color].tolist()
                # it will be replaced with the background color
                colors = [Masks.normalize_color((0.0, 0.0, 0.0, 0.0))]
                colors += [Masks.normalize_color(d[ll]) for ll in levels]
                c = np.concatenate(colors, axis=0)
                _legend = []
                for cat, col in zip(categories, cmap.colors):
                    _legend.append(
                        matplotlib.patches.Patch(facecolor=col, edgecolor=col, label=cat)
                    )
                return c
            elif type(color) == str and color == "random":
                a = np.random.rand(n, 3)
                b = np.ones(len(a)).reshape(-1, 1) * 1.0
                c = np.concatenate((a, b), axis=1)
                return c
            elif type(color) == str:
                a = matplotlib.colors.to_rgba(color)
                a = Masks.normalize_color(a)
                b = np.tile(a, (n, 1))
                return b
            elif type(color) == list or type(color) == np.ndarray:
                try:
                    a = matplotlib.colors.to_rgba(color)
                    b = [Masks.normalize_color(a)] * n
                    c = np.concatenate(b, axis=0)
                    return c
                except ValueError:
                    # it will be replaced with the background color
                    b = [Masks.normalize_color((0.0, 0.0, 0.0, 0.0))]
                    for c in color:
                        d = matplotlib.colors.to_rgba(c)
                        b.append(Masks.normalize_color(d))
                    e = np.concatenate(b, axis=0)
                    if len(e) != n:
                        raise ValueError(
                            "number of colors must match the number of elements in obs"
                        )
                    return e
            else:
                raise ValueError(f"invalid way of specifying the color: {color}")

        fill_color_array = get_color_array_rgba(fill_colors)
        outline_colors_array = get_color_array_rgba(outline_colors)
        background_color = Masks.normalize_color(matplotlib.colors.to_rgba(background_color))
        fill_color_array[0] = background_color
        outline_colors_array[0] = background_color
        for a in [fill_color_array, outline_colors_array]:
            Masks.apply_alpha(a, alpha=alpha)

        if ax is None:
            plt.figure()
            axs = plt.gca()
        else:
            axs = ax
        ############# calling back the plotting function
        self._plot(
            fill_color_array=fill_color_array,
            outline_colors_array=None if outline_colors is None else outline_colors_array,
            ax=axs,
        )

        if plotting_a_category:
            if show_title:
                axs.set_title(title)
            if show_legend:
                axs.legend(
                    handles=_legend,
                    frameon=False,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.1),
                    ncol=len(_legend),
                )
        # when the plot is invoked with my_fov.masks.plot(), then self.parentdataset._adjust_plot_lims() is called
        # once, when the plot is invoked with my_fov.plot(), then self.parentdataset._adjust_plot_lims() is called
        # twice (one now and one in the code for plotting FieldOfView subclasses). This should not pose a problem
        if self.parentdataset is not None:
            self.parentdataset._adjust_plot_lims(ax=axs)
        if ax is None:
            plt.show()


class ShapeMasks(Masks, MutableMapping):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        masks_centers: Optional[np.array] = None,
        masks_radii: Optional[Union[float, np.array]] = None,
        masks_shape: Optional[Literal["circle", "square"]] = None,
        masks_labels: Optional[list[str]] = None,
    ):
        super().__init__(backing=backing)

        self._masks_centers: Optional[np.ndarray] = None
        self._masks_radii: Optional[np.ndarray] = None
        self._masks_shape: Optional[SpotShape] = None
        self._masks_labels: Optional[list[str]] = None

        # radius is either one number, either a 1d vector either the same shape as the centers

        if self.is_backed:
            if (
                masks_centers is not None
                or masks_radii is not None
                or masks_shape is not None
                or masks_labels is not None
            ):
                raise ValueError("attempting to specify masks for a non-empty backing store")
            else:
                self._masks_centers = backing["masks_centers"]
                self._masks_radii = backing["masks_radii"]
                assert self._masks_centers.shape == self._masks_radii.shape
                s = _get_hdf5_attribute(backing.attrs, "masks_shape")
                self._masks_shape = SpotShape[s]
        else:
            if masks_shape is not None or masks_radii is not None:
                assert masks_centers is not None and masks_radii is not None
                # setting self._masks_shape
                if masks_shape is None:
                    self._masks_shape = SpotShape.circle
                else:
                    self._masks_shape = SpotShape[masks_shape]

                # setting self._masks_centers
                assert masks_centers is not None and masks_radii is not None
                assert len(masks_centers.shape) == 2
                n = masks_centers.shape[0]
                d = masks_centers.shape[1]
                assert d in [2, 3]
                self._masks_centers = masks_centers

                # setting self._masks.radii
                if isinstance(masks_radii, float):
                    self._masks_radii = np.ones_like(self._masks_centers) * masks_radii
                else:
                    assert isinstance(masks_radii, np.ndarray)
                    assert len(masks_radii) == n
                    if len(masks_radii.shape) in [0, 1]:
                        x = masks_radii.reshape((-1, 1))
                        self._masks_radii = np.tile(x, (1, d))
                    elif len(masks_radii.shape) == 2:
                        self._masks_radii = masks_radii
                    else:
                        raise ValueError()
                assert self._masks_radii.shape == self._masks_centers.shape

                # setting self._masks_labels
                if masks_labels is not None:
                    assert len(masks_labels) == n
                    self._masks_labels = masks_labels
            else:
                self._masks_shape = SpotShape.circle
                self._masks_centers = np.zeros([[]])
                self._masks_radii = np.zeros([[]])
            self.update_obs_from_masks()

    def update_obs_from_masks(self):
        # if the dataframe is not empty
        if self._obs is not None and len(self._obs.columns) != 0:
            raise ValueError(
                "replacing the old obs is only performed when obs is an empty DataFrame or it is None"
            )
        if self._masks_centers is None:
            raise ValueError("no mask data has been specified")
        if self._masks_labels is not None:
            mask_df = pd.DataFrame(data=dict(original_labels=self._masks_labels))
        else:
            mask_df = pd.DataFrame(index=range(len(self._masks_centers)))
        self._obs = mask_df

    @property
    def ndim(self):
        assert self._masks_centers is not None
        assert len(self._masks_centers.shape) == 2
        return self._masks_centers.shape[1]

    def __getitem__(self, key) -> Polygon:
        raise NotImplementedError()
        # if self.is_backed:
        #     return (self.backing[key]["center"], self.backing[key]["radius"])
        # else:
        #     return self._data[key]

    def __setitem__(self, key: str, value: tuple[tuple[float], float]):
        raise NotImplementedError()
        # if self.ndim is not None and self.ndim != len(value[0]):
        #     raise ValueError(f"value must have dimensionality {self.ndim}, but has {len(value[0])}")
        # if self.is_backed:
        #     grp = self.backing.create_group(key)
        #     grp.create_dataset("center", data=value[0])
        #     grp.create_dataset("radius", data=np.array([value[1]]))
        # else:
        #     self._data[key] = value

    def __delitem__(self, key: str):
        raise NotImplementedError()
        # if self.is_backed:
        #     del self.backing[key]
        # else:
        #     del self._data[key]

    def __len__(self):
        assert self._masks_centers is not None
        return len(self._masks_centers)
        # if self.is_backed:
        #     return len(self.backing)
        # else:
        #     return len(self._data)

    def __contains__(self, item):
        raise NotImplementedError()
        # if self.is_backed:
        #     return item in self.backing
        # else:
        #     return item in self._data

    def __iter__(self):
        raise NotImplementedError()
        # if self.is_backed:
        #     return iter(self.backing)
        # else:
        #     return iter(self._data)

    @property
    def _untransformed_bounding_box(self) -> BoundingBox:
        ##
        extended_coords = np.concatenate(
            [
                self._masks_centers[...] + self._masks_radii[...],
                self._masks_centers[...] - self._masks_radii[...],
            ],
            axis=0,
        )
        x_min, y_min = np.min(extended_coords, axis=0)
        x_max, y_max = np.max(extended_coords, axis=0)
        bb = BoundingBox(x0=x_min, x1=x_max, y0=y_min, y1=y_max)
        return bb

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

    def _set_backing(self, grp: Optional[h5py.Group]):
        super()._set_backing(grp)
        if grp is not None:
            assert isinstance(grp, h5py.Group)
            # self._backing should be reassigned from one of the caller functions (set_backing from BackableObject),
            # but to be safe let's set it to None explicity here
            self._backing = None
            self._write(grp)
        else:
            print("who is calling me?")
            assert self.is_backed

    def _write(self, grp: h5py.Group):
        super()._write(grp)
        grp.create_dataset("masks_centers", data=self._masks_centers)
        grp.create_dataset("masks_radii", data=self._masks_radii)
        pass

    def _write_attributes_impl(self, grp: h5py.Group):
        super()._write_attributes_impl(grp)
        grp.attrs["masks_shape"] = self._masks_shape.name

    def accumulate_features(self, x: Union["Raster", "Regions"]):
        # TODO:
        raise NotImplementedError()

    def extract_tiles(self, raster: "Raster", tile_dim: int):
        ome = raster.X

        raise NotImplementedError()

    def _plot(
        self,
        fill_color_array: np.ndarray,
        outline_colors_array: Optional[np.ndarray],
        ax: matplotlib.axes.Axes,
    ):
        if self._masks_shape != SpotShape.circle:
            raise NotImplementedError()
        patches = []
        for i in range(len(self)):
            xy = self._masks_centers[i]
            xy = self.anchor.transform_coordinates(xy)
            radius = self._masks_radii[i]
            radius /= self.anchor.scale_factor
            patch = matplotlib.patches.Ellipse(xy, width=2 * radius[0], height=2 * radius[1])
            patches.append(patch)
        collection = matplotlib.collections.PatchCollection(patches)
        # TODO: ignoring the background color for the moment, implement it
        collection.set_facecolor(fill_color_array[1:, :])
        if outline_colors_array is None:
            # need to plot always something otherwise the spot size is smaller
            collection.set_edgecolor(fill_color_array[1:, :])
        else:
            collection.set_edgecolor(outline_colors_array[1:, :])

        ax.set_aspect("equal")
        # autolim is set to False because it's our job to compute the lims considering the bounding
        # boxes of other fovs
        ax.add_collection(collection, autolim=False)

    def __repr__(self):
        return super().__repr__(mask_type="shape masks")


class PolygonMasks(Masks, MutableMapping):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        masks: Optional[dict[str, Union[np.ndarray, Polygon]]] = None,
        masks_labels: Optional[list[str]] = None,
    ):
        super().__init__(backing)

        self._data = {}

        # TODO: define private variables
        # TODO: check lenghts are ok

        if self.is_backed:
            if masks is not None or masks_labels is not None:
                raise ValueError("attempting to specify masks for a non-empty backing storage")
            else:
                # TODO: get stuff from the storage
                pass
        else:
            # TODO: save stuff to instance variables, if those are empty, specify a set of default empty masks
            pass

        # if masks is not None:
        #     if self.is_backed and len(self.backing) > 0:
        #         raise ValueError("trying to set masks on a non-empty backing store")
        #     for key, mask in masks:
        #         if isinstance(mask, Polygon):
        #             ndim = 3 if mask.has_z else 2
        #         else:
        #             ndim = mask.ndim
        #         if self.ndim is None:
        #             self.ndim = ndim
        #         if ndim != self.ndim:
        #             raise ValueError("all masks must have the same dimensionality")
        #
        #         self[key] = mask

    @property
    def ndim(self):
        # TODO:
        raise NotImplementedError()

    def __getitem__(self, key) -> Polygon:
        raise NotImplementedError()
        # if self.is_backed:
        #     return Polygon(self.backing[key][:])
        # else:
        #     return self._data[key]

    def __setitem__(self, key: str, value: Union[np.ndarray, Polygon]):
        raise NotImplementedError()
        # if self.is_backed:
        #     if isinstance(value, Polygon):
        #         value = np.asarray(value.exterior.coords)
        #     else:
        #         value = np.asarray(value)
        #     if self.ndim is not None and self.ndim != value.ndim:
        #         raise ValueError(
        #             f"value must have dimensionality {self.ndim}, but has {value.ndim}"
        #         )
        #     self.backing.create_dataset(key, data=value, compression="gzip", compression_opts=9)
        # else:
        #     if not isinstance(value, Polygon):
        #         ndim = value.ndim
        #         value = Polygon(value)
        #     else:
        #         ndim = 3 if value.has_z else 2
        #     if self.ndim is not None and self.ndim != ndim:
        #         raise ValueError(f"value must have dimensionality {self.ndim}, but has {ndim}")
        #     self._data[key] = value

    def __delitem__(self, key: str):
        raise NotImplementedError()
        # if self.is_backed:
        #     del self.backing[key]
        # else:
        #     del self._data[key]

    def __len__(self):
        # TODO:
        raise NotImplementedError()

    def __contains__(self, item):
        raise NotImplementedError()
        # if self.is_backed:
        #     return item in self.backing
        # else:
        #     return item in self._data

    def __iter__(self):
        raise NotImplementedError()
        # if self.is_backed:
        #     return iter(self.backing)
        # else:
        #     return iter(self._data)

    @property
    def _untransformed_bounding_box(self) -> BoundingBox:
        raise NotImplementedError()
        return dict()

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
        # TODO:
        raise NotImplementedError()
        # super()._set_backing(value)
        # if value is None and self.backed:
        #     for k, v in self.backing.items():
        #         self._data[k] = Polygon(v[:])
        # elif value is not None:
        #     self._write(value)
        #     self._data.clear()

    def _write(self, obj: h5py.Group):
        # TODO:
        raise NotImplementedError()
        # for k, v in self._data.items():
        #     obj.create_dataset(k, data=v.exterior.coords, compression="gzip", compression_opts=9)


class MeshMasks(Masks, MutableMapping):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        masks: Optional[dict[str, Union[Trimesh, tuple[np.ndarray, np.ndarray]]]] = None,
    ):
        super().__init__(backing)
        self._data = {}
        if masks is not None:
            if self.is_backed and len(self.backing) > 0:
                raise ValueError("trying to set masks on a non-empty backing store")
            self.update(masks)

    @property
    def ndim(self):
        return 3

    def __getitem__(self, key) -> Trimesh:
        if self.is_backed:
            return Trimesh(
                vertices=self.backing[key]["vertices"][()], faces=self.backing[key]["faces"][()]
            ).fix_normals()
        else:
            return self._data[key]

    def __setitem__(self, key: str, value: Union[Trimesh, tuple[np.ndarray, np.ndarray]]):
        if self.is_backed:
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
        if self.is_backed:
            del self.backing[key]
        else:
            del self._data[key]

    def __len__(self):
        if self.is_backed:
            return len(self.backing)
        else:
            return len(self._data)

    def __contains__(self, item):
        if self.is_backed:
            return item in self._backing
        else:
            return item in self._data

    def __iter__(self):
        if self.is_backed:
            return iter(self.backing)
        else:
            return iter(self._data)

    @property
    def _untransformed_bounding_box(self) -> BoundingBox:
        raise NotImplementedError()
        return dict()

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
        # why is obj not used in the following? Probably self.backing is to be replaced with obj
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
        px_dimensions: Optional[np.ndarray] = None,
        px_distance: Optional[np.ndarray] = None,
    ):
        super().__init__(backing=backing)
        self._mask = None

        if self.is_backed:
            if mask is not None:
                if self.backing.size > 0:
                    raise ValueError("attempting to set masks on a non-empty backing store")
                if mask.ndim < 2 or mask.ndim > 3:
                    raise ValueError("masks must have 2 or 3 dimensions")
                if not np.issubdtype(mask.dtype, np.unsignedinteger):
                    raise ValueError("masks must have an unsigned integer dtype")
                self._mask = mask
            else:
                self._mask = self.backing["imagemask"][...]
            self._px_distance = _get_hdf5_attribute(backing.attrs, "px_distance")
            self._px_dimensions = _get_hdf5_attribute(backing.attrs, "px_dimensions")
        else:
            if mask is not None:
                self._mask = mask
            else:
                if shape is None or dtype is None:
                    raise ValueError("if masks is None shape and dtype must be given")
                if len(shape) < 2 or len(shape) > 3:
                    raise ValueError("shape must have 2 or 3 dimensions")
                if not np.issubdtype(dtype, np.unsignedinteger):
                    raise ValueError("dtype must be an unsigned integer type")
                self._mask = np.zeros(shape=shape, dtype=dtype)
            self.update_obs_from_masks()
            self._px_dimensions = px_dimensions
            self._px_distance = px_distance
            if self._px_dimensions is not None:
                self._px_dimensions = np.asarray(self._px_dimensions).squeeze()
                if self._px_dimensions.shape[0] != self.ndim or self._px_dimensions.ndim != 1:
                    raise ValueError("pixel_size dimensionality is inconsistent with X")
            if self._px_distance is not None:
                self._px_distance = np.asarray(self._px_distance).squeeze()
                if self._px_distance.shape[0] != self.ndim or self._px_distance.ndim != 1:
                    raise ValueError("pixel_distance dimensionality is inconsistent with X")

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def px_dimensions(self) -> np.ndarray:
        if self._px_dimensions is None:
            return np.ones(self.ndim, np.uint8)
        else:
            return self._px_dimensions

    @property
    def px_distance(self) -> np.ndarray:
        if self._px_distance is None:
            return self.px_dimensions
        else:
            return self._px_distance

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    # TODO: this code is almost identical to the one in raster.py, do something to avoid redundancy
    @property
    def _untransformed_bounding_box(self) -> BoundingBox:
        assert self.ndim in [2, 3]
        if self.ndim == 3:
            raise NotImplementedError()
        h, w = self.data.shape[:2]
        px_dimensions = self.px_dimensions
        px_distance = self.px_distance
        actual_h = float(h) * px_dimensions[0] + (h - 1) * (px_distance[0] - 1)
        actual_w = float(w) * px_dimensions[1] + (w - 1) * (px_distance[1] - 1)
        bounding_box = BoundingBox(x0=0.0, y0=0.0, x1=actual_w, y1=actual_h)
        return bounding_box

    @property
    def data(self) -> Union[np.ndarray, h5py.Dataset]:
        if self.is_backed:
            return self.backing["imagemask"][...]
        else:
            return self._mask

    def __len__(self):
        return np.max(self.data)

    # flake8: noqa: C901
    def __getitem__(self, key):
        raise NotImplementedError()
        # if not np.issubdtype(type(key), np.integer):
        #     raise TypeError("key must be an integer")
        # if key >= 0:
        #     coords = np.where(self.data[()] == key + 1)
        #     if any(c.size == 0 for c in coords):
        #         raise KeyError(key)
        #     boundaries = []
        #     for i, c in enumerate(coords):
        #         min, max = np.min(c), np.max(c) + 1
        #         if min > 0:
        #             min -= 1
        #         if max <= self.data.shape[i]:
        #             max += 1
        #         boundaries.append(slice(min, max))
        #     cmask = self.data[tuple(boundaries)]
        #     cmask[cmask != key + 1] = 0
        #     if self.ndim == 2:
        #         contour = find_contours(cmask, fully_connected="high")
        #         for c in contour:
        #             for d in range(self.ndim):
        #                 c[:, d] += boundaries[d].start
        #         return (
        #             Polygon(contour[0][:, ::-1])
        #             if len(contour) == 1
        #             else [Polygon(c[:, ::-1]) for c in contour]
        #         )
        #         # TODO: scale by px_size and px_distance from parentdataset
        #     else:
        #         vertices, faces, normals, _ = marching_cubes(cmask, allow_degenerate=False)
        #         for d in range(self.ndim):
        #             vertices[:, d] += boundaries[d].start
        #             faces[:, d] += boundaries[d].start
        #         return Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        #         # TODO: scale by px_size and px_distance from parentdataset

    @staticmethod
    def _encodingtype():
        return "masks-raster"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, grp: Optional[h5py.Group]):
        super()._set_backing(grp)
        if grp is not None:
            assert isinstance(grp, h5py.Group)
            # self._backing should be reassigned from one of the caller functions (set_backing from BackableObject),
            # but to be safe let's set it to None explicity here
            self._backing = None
            self._write(grp)
            # self._mask = None
        else:
            print("who is calling me?")
            assert self.is_backed

    def _write(self, grp: h5py.Group):
        grp.create_dataset("imagemask", data=self._mask, compression="gzip", compression_opts=9)

    def _write_attributes_impl(self, obj: Union[h5py.Dataset, h5py.Group]):
        super()._write_attributes_impl(obj)
        if self._px_distance is not None:
            obj.attrs["px_distance"] = self._px_distance
        if self._px_dimensions is not None:
            obj.attrs["px_dimensions"] = self._px_dimensions

    def __repr__(self):
        return super().__repr__(mask_type="raster masks")

    # 0 is used as background label
    def update_obs_from_masks(self):
        # if the dataframe is not empty
        if self._obs is not None and len(self._obs.columns) != 0:
            raise ValueError(
                "replacing the old obs is only performed when obs is an empty DataFrame or it is None"
            )
        if self.data is None:
            raise ValueError("no mask data has been specified")
        m = self.data
        assert np.all(m >= 0)
        unique_masks = np.unique(m).tolist()
        # remove eventual background label
        unique_masks = [u for u in unique_masks if u != 0]
        mask_df = pd.DataFrame(data=dict(original_labels=unique_masks))
        self._obs = mask_df

    def _plot(
        self,
        fill_color_array: np.ndarray,
        outline_colors_array: Optional[np.ndarray],
        ax: matplotlib.axes.Axes,
    ):
        original_labels = self.obs["original_labels"].to_numpy()
        original_labels = np.insert(original_labels, 0, 0)
        contiguous_labels = np.arange(len(original_labels))
        x = self.data
        if not np.all(original_labels == contiguous_labels):
            # probably not the most efficient, but probably also fine
            lut = np.zeros(max(original_labels) + 1, dtype=np.int)
            for i, o in enumerate(original_labels):
                lut[o] = i
            x = lut[self.data]
        bb = self.bounding_box
        extent = [bb.x0, bb.x1, bb.y0, bb.y1]
        ax.imshow(fill_color_array[x], interpolation="none", extent=extent, origin="lower")

        if outline_colors_array is not None:
            for i, c in enumerate(contiguous_labels):
                outline_color = outline_colors_array[i]
                boolean_mask = x == c
                contours = skimage.measure.find_contours(boolean_mask, 0.7)
                for contour in contours:
                    contour = np.fliplr(contour)
                    transformed = self.anchor.transform_coordinates(contour)
                    ax.plot(transformed[:, 0], transformed[:, 1], linewidth=1, color=outline_color)

    def compute_centers(self):
        masks = self.data
        masks = masks.astype(np.uint32)
        ome = np.require(np.zeros_like(masks)[..., np.newaxis], requirements=["C"])
        vigra_ome = vigra.taggedView(ome, "xyc")
        ##
        features = vigra.analysis.extractRegionFeatures(
            vigra_ome,
            labels=masks,
            ignoreLabel=0,
            features=["RegionCenter"],
        )
        ##
        features = {k: v for k, v in features.items()}
        masks_with_obs = copy.copy(self)
        original_labels = masks_with_obs.obs["original_labels"].to_numpy()
        x = features["RegionCenter"][original_labels, 1]
        y = features["RegionCenter"][original_labels, 0]
        # original_labels = self.obs["original_labels"].to_numpy()
        # return original_labels, x, y
        return x, y

    def accumulate_features(self, fov: Union["Raster", "Regions"]):
        from spatialmuon.datatypes.regions import Regions

        if type(fov) == "Regions":
            raise NotImplementedError()
        x = fov.X[...]
        x = x.astype(np.float32)
        if type(fov) == "Raster" and len(x.shape) != 3:
            raise NotImplementedError()
        ome = np.require(x, requirements=["C"])
        vigra_ome = vigra.taggedView(ome, "xyc")
        masks = self.data
        masks = masks.astype(np.uint32)
        ##
        features = vigra.analysis.extractRegionFeatures(
            vigra_ome,
            labels=masks,
            ignoreLabel=0,
            features=["Count", "Maximum", "Mean", "Sum", "Variance", "RegionCenter"],
        )
        ##
        features = {k: v for k, v in features.items()}
        masks_with_obs = copy.copy(self)
        original_labels = masks_with_obs.obs["original_labels"].to_numpy()
        if (
            "region_center_x" not in masks_with_obs.obs
            and "region_center_y" not in masks_with_obs.obs
        ):
            masks_with_obs.obs["region_center_x"] = features["RegionCenter"][original_labels, 1]
            masks_with_obs.obs["region_center_y"] = features["RegionCenter"][original_labels, 0]
        if "count" not in masks_with_obs.obs:
            masks_with_obs.obs["count"] = features["Count"][original_labels]
        d = {}
        for key in ["Maximum", "Mean", "Sum", "Variance"]:
            regions = Regions(
                backing=None,
                X=features[key][original_labels, :],
                var=fov.var,
                masks=copy.copy(masks_with_obs),
                coordinate_unit=fov.coordinate_unit,
            )
            d[key.lower()] = regions
        return d

    def extract_tiles(self, raster: "Raster", tile_dim: int):
        DEBUG_WITH_PLOTS = False
        mask_labels = set(self.obs["original_labels"].to_list())
        ome = raster.X
        # here I need to find the bounding box from the masks and extract the pixels below
        real_labels = set(np.unique(self.data).tolist())
        if 0 in real_labels:
            real_labels.remove(0)
        assert mask_labels == real_labels
        extracted_omes = []
        extracted_masks = []
        z_centers = []
        origins = []
        for i, mask_label in tqdm(
            zip(range(len(mask_labels)), mask_labels),
            desc="extracting tiles",
            total=len(mask_labels),
        ):
            masks = self.data
            # center = xy[i, :]
            # compute the bounding box of the mask
            z = masks == mask_label
            z_center = center_of_mass(z)
            z_centers.append(np.array((z_center[1], z_center[0])))
            l = tile_dim
            # r = (l - 1) / 2
            # one pixel is lost but this makes computation easier
            r = math.floor(l / 2)
            if False:
                p0 = np.sum(z, axis=0)
                p1 = np.sum(z, axis=1)
                (w0,) = np.where(p0 > 0)
                (w1,) = np.where(p1 > 0)
                a0 = w0[0]
                b0 = w0[-1] + 1
                a1 = w1[0]
                b1 = w1[-1] + 1
            else:
                a0 = math.floor(z_center[1] - r)
                origin_x = a0
                a0 = max(0, a0)
                b0 = math.ceil(z_center[1] + r)
                b0 = min(b0, z.shape[1])
                a1 = math.floor(z_center[0] - r)
                origin_y = a1
                a1 = max(0, a1)
                b1 = math.ceil(z_center[0] + r)
                b1 = min(b1, z.shape[0])
                # print(f'[{a1}:{b1}, {a0}:{b0}]')
            # TODO: y this is empty when the tile around the mask does not contain any point of the mask and this
            #  leads to an exception. Handle this gracefully and show a warning
            y = z[a1:b1, a0:b0]
            if DEBUG_WITH_PLOTS:
                center_debug = np.array(center_of_mass(masks == mask_label))
                plt.figure(figsize=(20, 20))
                plt.imshow(z)
                plt.scatter(z_center[1], z_center[0], color="green", s=6)
                plt.scatter(center_debug[1], center_debug[0], color="red", s=2)
                plt.show()
                assert np.allclose(z_center, center_debug)

            y_center = np.array(center_of_mass(y))
            if DEBUG_WITH_PLOTS:
                plt.figure()
                plt.imshow(y)
                plt.scatter(y_center[1], y_center[0], color="black", s=1)
                plt.show()
            square_ome = np.zeros((l, l, ome.shape[2]))
            square_mask = np.zeros((l, l))

            def get_coords_for_padding(des_r, src_shape, src_center):
                des_l = 2 * des_r + 1

                def f(src_l, src_c):
                    a = src_c - des_r
                    b = src_c + des_r
                    if a < 0:
                        c = -a
                        a = 0
                    else:
                        c = 0
                    if b > src_l:
                        b = src_l
                    src_a = a
                    src_b = b
                    des_a = c
                    des_b = des_a + b - a
                    return src_a, src_b, des_a, des_b

                src0_a, src0_b, des0_a, des0_b = f(src_shape[0], int(src_center[0]))
                src1_a, src1_b, des1_a, des1_b = f(src_shape[1], int(src_center[1]))
                return src0_a, src0_b, src1_a, src1_b, des0_a, des0_b, des1_a, des1_b

            # fmt: off
            (
                src0_a, src0_b, src1_a, src1_b,
                des0_a, des0_b, des1_a, des1_b,
            ) = get_coords_for_padding(r, y.shape, y_center)
            # fmt: on

            square_ome[des0_a:des0_b, des1_a:des1_b, :] = ome[a1:b1, a0:b0, :][
                src0_a:src0_b, src1_a:src1_b, :
            ]
            square_mask[des0_a:des0_b, des1_a:des1_b] = y[src0_a:src0_b, src1_a:src1_b]

            if DEBUG_WITH_PLOTS:
                plt.figure()
                plt.imshow(square_mask)
                plt.scatter(r, r, color="blue", s=1)
                plt.show()
            #                     f5_out[f'{split}/omes/{k}'] = ome[a1: b1, a0: b0]
            #                     f5_out[f'{split}/masks/{k}'] = y
            extracted_omes.append(square_ome)
            extracted_masks.append(square_mask)
            origins.append(np.array([origin_x, origin_y]))
            if DEBUG_WITH_PLOTS:
                if i >= 4:
                    DEBUG_WITH_PLOTS = False
                    RECAP_PLOT = True
        if "RECAP_PLOT" in locals():
            for n in range(min(len(extracted_omes), 10)):
                plt.figure()
                x = extracted_omes[n][:, :, 0]
                plt.imshow(x)
                u = np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 1.0],
                    ]
                )
                mask = extracted_masks[n]
                plt.imshow(u[mask.astype(int)])
                plt.show()
        return extracted_omes, extracted_masks, z_centers, origins


if __name__ == "__main__":
    x = np.array(range(100), dtype=np.uint).reshape(10, 10)

    rm = RasterMasks(backing=None, mask=x)
    rm.update_obs_from_masks()
