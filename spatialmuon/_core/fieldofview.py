from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Optional, Union, Literal, Dict
import warnings
import matplotlib.pyplot as plt

import numpy as np
import h5py
from shapely.geometry import Polygon
from trimesh import Trimesh
from anndata._io.utils import read_attribute, write_attribute
from anndata.utils import make_index_unique
import pandas as pd

from spatialmuon._core.backing import BackableObject
from spatialmuon._core.anchor import Anchor
from spatialmuon._core.bounding_box import BoundingBoxable, BoundingBox

# from .image import Image
from ..utils import _read_hdf5_attribute, _get_hdf5_attribute, UnknownEncodingException


class FieldOfView(BackableObject, BoundingBoxable):
    _datatypes = {}

    @classmethod
    def _load_datatypes(cls):
        if len(cls._datatypes) == 0:
            datatypes = entry_points()["spatialmuon.datatypes"]
            for ep in datatypes:
                klass = ep.load()
                cls._datatypes[klass._encodingtype()] = klass

    def __new__(cls, *, backing: Optional[h5py.Group] = None, **kwargs):
        if backing is not None:
            fovtype = _read_hdf5_attribute(backing.attrs, "encoding-type")
            cls._load_datatypes()
            if fovtype in cls._datatypes:
                klass = cls._datatypes[fovtype]
                return super(cls, klass).__new__(klass)
            else:
                raise UnknownEncodingException(fovtype)
        else:
            return super().__new__(cls)

    # flake8: noqa: C901
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        var: Optional[pd.DataFrame] = None,
        uns: Optional[dict] = None,
        coordinate_unit: Optional[str] = None,
        anchor: Optional[Anchor] = None,
    ):
        super().__init__(backing)
        if backing is not None:
            if (
                var is not None
                or uns is not None
                or coordinate_unit is not None
                or anchor is not None
            ):
                raise ValueError("attempting to specify properties for a non-empty backing store")
            else:
                self.var = read_attribute(backing["var"])
                # self._var = read_attribute(backing["var"])
                # the function writing self.uns to disk in from the anndata package, and by default doesn't store uns
                # if it's value is {}, so we have to check for existence
                if "uns" in self.backing:
                    self.uns = read_attribute(self.backing["uns"])
                else:
                    self.uns = {}
                self.coordinate_unit = _get_hdf5_attribute(
                    self.backing.attrs, "coordinate_unit", None
                )
                self.anchor = Anchor(backing=backing["anchor"])
        else:
            if var is not None:
                self.var = var
            else:
                self.var = pd.DataFrame()
                # self._var = pd.DataFrame()

            if uns is not None:
                self.uns = uns
            else:
                self.uns = {}

            if coordinate_unit is not None:
                self.coordinate_unit = coordinate_unit
            else:
                self.coordinate_unit = "units"

            if anchor is not None:
                self.anchor = anchor
            else:
                # in the new version of the code the function update_n_dim_in_anchor called by FieldOfView subclasses
                # implies that ancor is not None, so this code should not be reached
                self.anchor = Anchor(self.ndim)
                warnings.warn("who called me?")
        # disabled by default since it is slowe
        self.compressed_storage = False

    # @property
    # def _backed_children(self) -> Dict[str, "BackableObject"]:
    #     assert self._anchor is not None
    #     return {'anchor': self._anchor}

    # in classes inherithing from FieldOfView, this function is the only thing called in __init__() before calling
    # super().__init__(). This becase the __init__() in FieldOfView needs to know which is the dimensionality of the
    # data (2D vs 3D), in order to initialize a default value for self.ancors, and this requires information
    # contained in the arguments passed to __init__() of the subclass (but not passed to the superclass). Then,
    # after __init__() from FieldOfView is executed, in FieldOfView the ndim property will only make use of
    # self.anchor, so subclasses can know their dimensionality by asking the superclass. An alternative approach
    # would be to implement ndim as an abstract method, but in this way, if super().__init__() is
    # called in the beginning of the __init__() of subclasses as standard in Python, then there is no way to ask the
    # subclass about ndim, because the subclass is not initialized yet
    def update_n_dim_in_anchor(self, ndim: Optional[int], backing, **kwargs) -> Optional[Anchor]:
        assert not (backing is not None and "anchor" in backing and "anchor" in kwargs)
        if backing is not None and "anchor" in backing:
            return None
        elif "anchor" in kwargs:
            return kwargs["anchor"]
        else:
            assert ndim is not None
            return Anchor(ndim=ndim)

    @property
    def ndim(self) -> int:
        return self.anchor.ndim

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @var.setter
    def var(self, new_var):
        self._var = new_var
        if not self._var.index.is_unique:
            warnings.warn(
                "Gene names are not unique. This will negatively affect indexing/subsetting. Making unique names..."
            )
            self._var.index = make_index_unique(self._var.index)

    @property
    def n_var(self) -> int:
        return self._var.shape[0]

    @property
    def anchor(self) -> Anchor:
        """A np.ndarray with an anchor/vector pair for alignment.
        Spatial information can be aligned to eachother in a m:n fashion. This
        is implemented in spatialmuon on the basis of an anchor point from which
        a vector extends that is aligned in a global coordinate system. All data
        shares this global coordinate system and aligns to eachother in it.

        """
        # assert self._anchor is not None
        # return self._anchor
        assert 'anchor' in self
        return self['anchor']

    @anchor.setter
    def anchor(self, new_anchor):
        self['anchor'] = new_anchor

    # def __getitem__(self, index):
    #     polygon_method = "discard"
    #     if not isinstance(index, tuple):
    #         if isinstance(index, str) or isinstance(index, list):
    #             genes = index
    #             mask = None
    #         else:
    #             mask = index
    #             genes = None
    #     else:
    #         mask = index[0]
    #         genes = index[1]
    #         if len(index) > 2:
    #             polygon_method = index[2]
    #     if mask == slice(None):
    #         mask = None
    #     if genes == slice(None):
    #         genes = None
    #
    #     return self._getitem(mask, genes, polygon_method)

    @abstractmethod
    def _getitem(
        self,
        mask: Optional[Union[Polygon, Trimesh]] = None,
        genes: Optional[Union[str, list[str]]] = None,
        polygon_method: Literal["discard", "project"] = "discard",
    ):
        pass

    def _write_attributes_impl(self, obj: h5py.Group):
        super()._write_attributes_impl(obj)
        if self.coordinate_unit is not None:
            obj.attrs["coordinate_unit"] = self.coordinate_unit

    def _write_impl(self, obj: h5py.Group):
        if self.compressed_storage:
            write_attribute(
                obj, "var", self._var, dataset_kwargs={"compression": "gzip", "compression_opts": 9}
            )
            write_attribute(
                obj, "uns", self.uns, dataset_kwargs={"compression": "gzip", "compression_opts": 9}
            )
        else:
            write_attribute(obj, "var", self._var)
            write_attribute(obj, "uns", self.uns)

    def _adjust_plot_lims(self, ax=None):
        if ax is None:
            ax = plt.gca()
        bb = self.bounding_box
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # hack to check if this is a newly created empty plot
        if xlim == (0.0, 1.0) and ylim == (0.0, 1.0):
            new_xlim = (bb.x0, bb.x1)
            new_ylim = (bb.y0, bb.y1)
        else:
            new_xlim = (min(xlim[0], bb.x0), max(xlim[1], bb.x1))
            new_ylim = (min(ylim[0], bb.y0), max(ylim[1], bb.y1))
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        # print(f'xlim = {xlim}, ylim = {ylim}')
        # print(f'new_xlim = {new_xlim}, new_ylim = {new_ylim}')
        # pass

    def set_lims_to_bounding_box(self, bb: BoundingBox = None, ax=None):
        if ax is None:
            ax = plt.gca()
        if bb is None:
            bb = self.bounding_box
        ax.set_xlim((bb.x0, bb.x1))
        ax.set_ylim((bb.y0, bb.y1))
