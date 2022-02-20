from __future__ import annotations

from collections.abc import MutableMapping
from abc import abstractmethod
from typing import Optional, Union, Literal, TYPE_CHECKING, Tuple, Dict

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.axes
import matplotlib.cm
import matplotlib.patches
import matplotlib.collections
import numpy as np
import h5py
import copy

import shapely.geometry
from shapely.geometry import Polygon
from trimesh import Trimesh
from skimage.measure import find_contours
from anndata._io.utils import read_attribute, write_attribute
import pandas as pd
import skimage.measure
import vigra
from functools import cached_property
from enum import Enum, auto

from spatialmuon._core.backing import BackableObject
from spatialmuon.utils import (
    _read_hdf5_attribute,
    UnknownEncodingException,
    _get_hdf5_attribute,
    ColorsType,
    ColorType,
    normalize_color,
    apply_alpha,
    handle_categorical_plot,
    get_color_array_rgba,
)
from spatialmuon._core.bounding_box import BoundingBoxable, BoundingBox

if TYPE_CHECKING:
    from spatialmuon import Raster, Regions


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
            if obs is not None:
                raise ValueError("attempting to specify masks for a non-empty backing store")
            self._obs = read_attribute(backing["obs"])
        else:
            if obs is not None:
                self.obs = obs
            else:
                self.obs = pd.DataFrame()

    @property
    def anchor(self):
        return self._parentdataset.anchor

    @anchor.setter
    def anchor(self, new_anchor):
        self._parentdataset.anchor = new_anchor

    @property
    def coordinate_unit(self):
        return self._parentdataset.coordinate_unit

    @coordinate_unit.setter
    def coordinate_unit(self, o):
        self._parentdataset.coordinate_unit = o

    # we don't provide a setter for ndim
    @property
    @abstractmethod
    def ndim(self):
        pass

    # @abstractmethod
    # def __getitem__(self, key):
    #     pass

    @abstractmethod
    def __len__(self):
        pass

    # @abstractmethod
    def update_obs_from_masks(self):
        pass

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @obs.setter
    def obs(self, new_obs):
        self._obs = new_obs
        self.obj_has_changed("obs")

    @property
    def n_obs(self) -> int:
        return self._obs.shape[0]

    @property
    @abstractmethod
    def untransformed_masks_centers(self):
        pass

    @property
    def transformed_masks_centers(self):
        return self.anchor.transform_coordinates(self.untransformed_masks_centers[...])

    # def _write_data(self, grp):
    def _write_attributes_impl(self, obj: Union[h5py.Dataset, h5py.Group]):
        super()._write_attributes_impl(obj)
        if self.has_obj_changed("obs"):
            if "geometry" in self._obs.columns:
                o = (self.obs.drop(self._obs.geometry.name, axis=1),)
            else:
                o = self.obs

            write_attribute(
                obj,
                "obs",
                o,
                dataset_kwargs={"compression": "gzip", "compression_opts": 9},
            )

    def __repr__(self, mask_type="masks"):
        repr_str = f"│   ├── {self.ndim}D {mask_type} with {self.n_obs} obs: {', '.join(self.obs)}"
        repr_str = "│   └──".join(repr_str.rsplit("│   ├──", 1))
        return repr_str

    @abstractmethod
    def crop(self, bounding_box: BoundingBox):
        pass

    @abstractmethod
    def subset_obs(self, indices: np.array, inplace: bool = False):
        pass

    @abstractmethod
    def _plot(
        self,
        fill_color_array: np.ndarray,
        outline_color_array: Optional[np.ndarray],
        ax: matplotlib.axes.Axes,
    ):
        pass

    # flake8: noqa: C901
    def plot(
        self,
        fill_colors: ColorsType = "random",
        outline_colors: ColorsType = None,
        background_color: Optional[Union[str, np.array]] = (0.0, 0.0, 0.0, 0.0),
        ax: matplotlib.axes.Axes = None,
        alpha: float = 1.0,
        show_title: bool = True,
        show_legend: bool = True,
        bounding_box: Optional[BoundingBox] = None,
        figsize: Optional[Tuple[int]] = None,
        categories_colors: Optional[Dict[str, ColorType]] = None
    ):
        if bounding_box is not None:
            # a copy that simplifies the code but is not really necessary, we could make crop work inplace
            masks_subset = self.clone()
            masks_subset = masks_subset.crop(bounding_box=bounding_box)
            ii = masks_subset.obs.index.to_numpy()
            if fill_colors is not None:
                fill_colors = fill_colors[ii]
            if outline_colors is not None:
                outline_colors = outline_colors[ii]
            masks_subset.plot(
                fill_colors=fill_colors,
                outline_colors=outline_colors,
                background_color=background_color,
                ax=ax,
                alpha=alpha,
                show_title=show_title,
                show_legend=show_legend,
                bounding_box=None,
            )
            return

        # adding the background
        n = len(self.obs) + 1

        fill_color_array, plotting_a_category, title, _legend = handle_categorical_plot(
            fill_colors, obs=self.obs, categories_colors=categories_colors
        )
        if not plotting_a_category:
            fill_color_array = get_color_array_rgba(fill_colors, n)

        outline_color_array, plotting_a_category, title, _legend = handle_categorical_plot(
            outline_colors, obs=self.obs, categories_colors=categories_colors
        )
        if not plotting_a_category:
            outline_color_array = get_color_array_rgba(outline_colors, n)

        background_color = normalize_color(matplotlib.colors.to_rgba(background_color))
        fill_color_array[0] = background_color
        outline_color_array[0] = background_color
        for a in [fill_color_array, outline_color_array]:
            apply_alpha(a, alpha=alpha)

        if ax is not None and figsize is not None:
            raise ValueError("ax and figsize cannot be both non-None")

        if ax is None:
            plt.figure(figsize=figsize)
            axs = plt.gca()
        else:
            axs = ax
        # ############ calling back the plotting function
        self._plot(
            fill_color_array=fill_color_array,
            outline_color_array=None if outline_colors is None else outline_color_array,
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
        # when the plot is invoked with my_fov.masks.plot(), then self._parentdataset._adjust_plot_lims() is called
        # once, when the plot is invoked with my_fov.plot(), then self._parentdataset._adjust_plot_lims() is called
        # twice (one now and one in the code for plotting FieldOfView subclasses). This should not pose a problem
        if self._parentdataset is not None:
            self._parentdataset._adjust_plot_lims(ax=axs)
        if ax is None:
            plt.show()

    def accumulate_features(self, fov: Union[Raster, Regions]):
        from spatialmuon.processing.accumulator import accumulate_features

        accumulate_features(masks=self, fov=fov)

    def extract_tiles(
        self, tile_dim_in_units: Optional[float] = None, tile_dim_in_pixels: Optional[float] = None
    ):
        from spatialmuon.processing.tiler import Tiles

        return Tiles(
            masks=self,
            raster=None,
            tile_dim_in_units=tile_dim_in_units,
            tile_dim_in_pixels=tile_dim_in_pixels,
        )


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
                self._masks_labels = _get_hdf5_attribute(backing.attrs, "masks_labels")
        else:
            if masks_centers is not None or masks_radii is not None:
                # setting self._masks_centers
                assert masks_centers is not None and masks_radii is not None
                assert len(masks_centers.shape) == 2
                n = masks_centers.shape[0]
                d = masks_centers.shape[1]
                assert d in [2, 3]
                self.untransformed_masks_centers = masks_centers

                # setting self._masks_shape
                if masks_shape is None:
                    self.masks_shape = SpotShape.circle
                else:
                    self.masks_shape = SpotShape[masks_shape]

                # setting self._masks.radii
                if isinstance(masks_radii, float):
                    self.untransformed_masks_radii = np.ones_like(self._masks_centers) * masks_radii
                else:
                    assert isinstance(masks_radii, np.ndarray)
                    assert len(masks_radii) == n
                    if len(masks_radii.shape) in [0, 1]:
                        x = masks_radii.reshape((-1, 1))
                        self.untransformed_masks_radii = np.tile(x, (1, d))
                    elif len(masks_radii.shape) == 2:
                        self.untransformed_masks_radii = masks_radii
                    else:
                        raise ValueError()
                assert (
                    self.untransformed_masks_radii.shape == self.untransformed_masks_centers.shape
                )

                # setting self._masks_labels
                if masks_labels is not None:
                    assert len(masks_labels) == n
                    self.masks_labels = masks_labels
                else:
                    self.masks_labels = list(map(str, range(n)))
            else:
                self.masks_shape = SpotShape.circle
                self.untransformed_masks_centers = np.zeros([[]])
                self.untransformed_masks_radii = np.zeros([[]])
                self.masks_labels = []
            self.update_obs_from_masks()

    def update_obs_from_masks(self):
        # if the dataframe is not empty
        if len(self.obs.columns) != 0:
            raise ValueError(
                "replacing the old obs is only performed when obs is an empty DataFrame or it is None"
            )
        if len(self.untransformed_masks_centers) == 0:
            raise ValueError("no mask data has been specified")
        if len(self.masks_labels) == 0:
            mask_df = pd.DataFrame(data=dict(original_labels=self.masks_labels))
        else:
            mask_df = pd.DataFrame(index=range(len(self.untransformed_masks_centers)))
        self.obs = mask_df

    # no setter for ndim
    @property
    def ndim(self):
        assert len(self.untransformed_masks_centers) > 0
        assert len(self.untransformed_masks_centers.shape) == 2
        return self.untransformed_masks_centers.shape[1]

    @property
    def masks_shape(self):
        return self._masks_shape

    @masks_shape.setter
    def masks_shape(self, o):
        self._masks_shape = o
        self.obj_has_changed("masks_shape")

    @property
    def untransformed_masks_centers(self):
        return self._masks_centers

    @untransformed_masks_centers.setter
    def untransformed_masks_centers(self, o):
        self._masks_centers = o
        self.obj_has_changed("masks_centers")

    @property
    def untransformed_masks_radii(self):
        return self._masks_radii

    @untransformed_masks_radii.setter
    def untransformed_masks_radii(self, o):
        self._masks_radii = o
        self.obj_has_changed("masks_radii")

    @property
    def transformed_masks_radii(self):
        radii = self.untransformed_masks_radii
        transformed = [self.anchor.transform_length(r) for r in radii]
        return transformed

    @property
    def masks_labels(self):
        return self._masks_labels

    @masks_labels.setter
    def masks_labels(self, o):
        self._masks_labels = o
        self.obj_has_changed("masks_labels")

    def __len__(self):
        assert len(self.untransformed_masks_centers) > 0
        return len(self.untransformed_masks_centers)

    # def __contains__(self, item):
    #     raise NotImplementedError()
    #     # if self.is_backed:
    #     #     return item in self.backing
    #     # else:
    #     #     return item in self._data

    # def __iter__(self):
    #     raise NotImplementedError()
    #     # if self.is_backed:
    #     #     return iter(self.backing)
    #     # else:
    #     #     return iter(self._data)

    def crop(self, bounding_box: BoundingBox):
        polygon = bounding_box.to_polygon()
        if self.masks_shape == SpotShape.rectangle:
            raise NotImplementedError()
        else:
            assert self.masks_shape == SpotShape.circle
            to_keep = []
            for i, (xy, r) in enumerate(
                zip(self.transformed_masks_centers, self.transformed_masks_radii)
            ):
                # shapely does not support ellipses
                max_r = np.max(r)
                p = shapely.geometry.Point(xy).buffer(max_r)
                if polygon.intersects(p):
                    to_keep.append(i)
            b = np.array(to_keep)
            o = self.subset_obs(indices=to_keep)
            return o

    def subset_obs(self, indices: np.array, inplace: bool = False):
        if inplace:
            raise NotImplementedError()
        else:
            o = self.clone()
            new_obs = o.obs.iloc[indices]
            o.obs = new_obs
            o.untransformed_masks_centers = o.untransformed_masks_centers[indices, :]
            o.untransformed_masks_radii = o.untransformed_masks_radii[indices, :]
            o.masks_labels = o.masks_labels[indices]
            return o

    @property
    def _untransformed_bounding_box(self) -> BoundingBox:
        ##
        extended_coords = np.concatenate(
            [
                self.untransformed_masks_centers[...] + self.untransformed_masks_radii[...],
                self.untransformed_masks_centers[...] - self.untransformed_masks_radii[...],
            ],
            axis=0,
        )
        x_min, y_min = np.min(extended_coords, axis=0)
        x_max, y_max = np.max(extended_coords, axis=0)
        bb = BoundingBox(x0=x_min, x1=x_max, y0=y_min, y1=y_max)
        return bb

    @staticmethod
    def _encodingtype():
        return "masks-shape"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _write_impl(self, grp: h5py.Group):
        super()._write_impl(grp)
        if self.has_obj_changed("masks_centers"):
            if "masks_centers" in grp:
                del grp["masks_centers"]
            grp.create_dataset("masks_centers", data=self.untransformed_masks_centers)
        if self.has_obj_changed("masks_radii"):
            if "masks_radii" in grp:
                del grp["masks_radii"]
            grp.create_dataset("masks_radii", data=self.untransformed_masks_radii)

    def _write_attributes_impl(self, grp: h5py.Group):
        super()._write_attributes_impl(grp)
        if self.has_obj_changed("masks_shape"):
            grp.attrs["masks_shape"] = self.masks_shape.name
        if self.has_obj_changed("masks_labels"):
            grp.attrs["masks_labels"] = self.masks_labels

    def _plot(
        self,
        fill_color_array: np.ndarray,
        outline_color_array: Optional[np.ndarray],
        ax: matplotlib.axes.Axes,
    ):
        if self.masks_shape != SpotShape.circle:
            raise NotImplementedError()
        patches = []
        for i in range(len(self)):
            xy = self.untransformed_masks_centers[i]
            xy = self.anchor.transform_coordinates(xy)
            radius = self.untransformed_masks_radii[i]
            radius /= self.anchor.scale_factor
            patch = matplotlib.patches.Ellipse(xy, width=2 * radius[0], height=2 * radius[1])
            patches.append(patch)
        collection = matplotlib.collections.PatchCollection(patches)
        # TODO: ignoring the background color for the moment, implement it
        collection.set_facecolor(fill_color_array[1:, :])
        if outline_color_array is None:
            # need to plot always something otherwise the spot size is smaller
            collection.set_edgecolor(fill_color_array[1:, :])
        else:
            collection.set_edgecolor(outline_color_array[1:, :])

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

    # no setter for ndim
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
    def centers(self):
        raise NotImplementedError()

    def crop(self, bounding_box: BoundingBox):
        raise NotImplementedError()

    def subset_obs(self, indices: np.array, inplace: bool = False):
        raise NotImplementedError()

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

    # def _set_backing(self, value: h5py.Group):
    #     # TODO:
    #     raise NotImplementedError()
    # super()._set_backing(value)
    # if value is None and self.backed:
    #     for k, v in self.backing.items():
    #         self._data[k] = Polygon(v[:])
    # elif value is not None:
    #     self._write_impl(value)
    #     self._data.clear()

    def _write_impl(self, obj: h5py.Group):
        super()._write_impl(obj)
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

    # no setter for ndim
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
    def centers(self):
        raise NotImplementedError()

    def crop(self, bounding_box: BoundingBox):
        raise NotImplementedError()

    def subset_obs(self, indices: np.array, inplace: bool = False):
        raise NotImplementedError()

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

    # def _set_backing(self, value: h5py.Group):
    #     raise NotImplementedError()
    #     super()._set_backing(value)
    #     if value is None and self.backed:
    #         for k in self.keys():
    #             self._data[k] = self[k]
    #     elif value is not None:
    #         self._write_impl(value)
    #         self._data.clear()

    def _write_impl(self, obj: h5py.Group):
        raise NotImplementedError()
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
        X: Optional[np.ndarray] = None,
        px_dimensions: Optional[np.ndarray] = None,
        px_distance: Optional[np.ndarray] = None,
    ):
        super().__init__(backing=backing)

        if self.is_backed:
            if X is not None or px_dimensions is not None or px_distance is not None:
                raise ValueError("attempting to specify masks for a non-empty backing store")

            self._X = self.backing["X"]
            self._px_distance = _get_hdf5_attribute(backing.attrs, "px_distance")
            self._px_dimensions = _get_hdf5_attribute(backing.attrs, "px_dimensions")
        else:
            assert X is not None
            if not np.issubdtype(X.dtype, np.unsignedinteger):
                raise ValueError("dtype must be an unsigned integer type")
            if X.ndim < 2 or X.ndim > 3:
                raise ValueError("masks must have 2 or 3 dimensions")
            self.X = X

            if px_dimensions is not None:
                self.px_dimensions = px_dimensions
                if len(self.px_dimensions.squeeze()) != self.ndim:
                    raise ValueError("px_dimensions dimensionality is inconsistent with X")
            else:
                self.px_dimensions = np.ones(self.ndim, np.uint8)

            if px_distance is not None:
                self.px_distance = px_distance
                if len(self.px_distance.squeeze()) != self.ndim:
                    raise ValueError("px_distance dimensionality is inconsistent with X")
            else:
                self.px_distance = np.ones(self.ndim, np.uint8)
            self.update_obs_from_masks()

    # no setter for ndim
    @property
    def ndim(self):
        return self.X.ndim

    @property
    def X(self) -> Union[np.ndarray, h5py.Dataset]:
        return self._X

    @X.setter
    def X(self, new_X):
        self._X = new_X
        self.obj_has_changed("X")

    @property
    def px_dimensions(self) -> np.ndarray:
        return self._px_dimensions

    @px_dimensions.setter
    def px_dimensions(self, o):
        self._px_dimensions = o
        self.obj_has_changed("px_dimensions")

    @property
    def px_distance(self) -> np.ndarray:
        return self._px_distance

    @px_distance.setter
    def px_distance(self, o):
        self._px_distance = o
        self.obj_has_changed("px_distance")

    @property
    def shape(self):
        return self.X.shape

    @property
    def dtype(self):
        return self.X.dtype

    # TODO: this code is almost identical to the one in raster.py, do something to avoid redundancy
    @property
    def _untransformed_bounding_box(self) -> BoundingBox:
        assert self.ndim in [2, 3]
        if self.ndim == 3:
            raise NotImplementedError()
        h, w = self.X.shape[:2]
        px_dimensions = self.px_dimensions
        px_distance = self.px_distance
        actual_h = float(h) * px_dimensions[0] + (h - 1) * (px_distance[0] - 1)
        actual_w = float(w) * px_dimensions[1] + (w - 1) * (px_distance[1] - 1)
        bounding_box = BoundingBox(x0=0.0, y0=0.0, x1=actual_w, y1=actual_h)
        return bounding_box

    def __len__(self):
        return np.max(self.X)

    # # flake8: noqa: C901
    # def __getitem__(self, key):
    #     raise NotImplementedError()
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

    def crop(self, bounding_box: BoundingBox):
        raise NotImplementedError()

    def subset_obs(self, indices: np.array, inplace: bool = False):
        raise NotImplementedError()

    @staticmethod
    def _encodingtype():
        return "masks-raster"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _write_impl(self, grp: h5py.Group):
        super()._write_impl(grp)
        if self.has_obj_changed("X"):
            if "X" in grp:
                del grp["X"]
            grp.create_dataset("X", data=self.X, compression="gzip", compression_opts=9)

    def _write_attributes_impl(self, obj: Union[h5py.Dataset, h5py.Group]):
        super()._write_attributes_impl(obj)
        if self.has_obj_changed("px_dimensions"):
            obj.attrs["px_dimensions"] = self.px_dimensions
        if self.has_obj_changed("px_distance"):
            obj.attrs["px_distance"] = self.px_distance

    def __repr__(self):
        return super().__repr__(mask_type="raster masks")

    # 0 is used as background label
    def update_obs_from_masks(self):
        # if the dataframe is not empty
        if len(self.obs.columns) != 0:
            raise ValueError(
                "replacing the old obs is only performed when obs is an empty DataFrame or it is None"
            )
        m = self.X[...]
        assert np.all(m >= 0)
        unique_masks = np.unique(m).tolist()
        # remove eventual background label
        unique_masks = [u for u in unique_masks if u != 0]
        mask_df = pd.DataFrame(data=dict(original_labels=unique_masks))
        self.obs = mask_df

    def _plot(
        self,
        fill_color_array: np.ndarray,
        outline_color_array: Optional[np.ndarray],
        ax: matplotlib.axes.Axes,
    ):
        original_labels = self.obs["original_labels"].to_numpy()
        original_labels = np.insert(original_labels, 0, 0)
        contiguous_labels = np.arange(len(original_labels))
        x = self.X[...]
        if not np.all(original_labels == contiguous_labels):
            # probably not the most efficient, but probably also fine
            lut = np.zeros(max(original_labels) + 1, dtype=np.int)
            for i, o in enumerate(original_labels):
                lut[o] = i
            x = lut[self.X]
        bb = self.bounding_box
        extent = [bb.x0, bb.x1, bb.y0, bb.y1]
        ax.imshow(fill_color_array[x], interpolation="none", extent=extent, origin="lower")

        if outline_color_array is not None:
            for i, c in enumerate(contiguous_labels):
                outline_color = outline_color_array[i]
                boolean_mask = x == c
                contours = skimage.measure.find_contours(boolean_mask, 0.7)
                for contour in contours:
                    contour = np.fliplr(contour)
                    contour += 0.5
                    transformed = self.anchor.transform_coordinates(contour)
                    ax.plot(transformed[:, 0], transformed[:, 1], linewidth=1, color=outline_color)

    @property
    def untransformed_masks_centers(self):
        # from a tuple of 1-dim vectors to a n x 2 tensor
        return np.vstack(self._centers).T

    @cached_property
    def _centers(self):
        masks = self.X[...]
        masks = masks.astype(np.uint32)
        ome = np.require(
            np.zeros_like(masks, dtype=np.float32)[..., np.newaxis], requirements=["C"]
        )
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
        masks_with_obs = self.clone()
        # masks_with_obs = copy.copy(self)
        original_labels = masks_with_obs.obs["original_labels"].to_numpy()
        x = features["RegionCenter"][original_labels, 1]
        y = features["RegionCenter"][original_labels, 0]
        # original_labels = self.obs["original_labels"].to_numpy()
        # return original_labels, x, y
        return x, y


if __name__ == "__main__":
    x = np.array(range(100), dtype=np.uint).reshape(10, 10)

    rm = RasterMasks(backing=None, X=x)
    rm.update_obs_from_masks()
