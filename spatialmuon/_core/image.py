from __future__ import annotations

from typing import Optional, Union, List

import numpy as np
import h5py

from .backing import BackableObject
from ..utils import _get_hdf5_attribute, _read_hdf5_attribute


class Image(BackableObject):
    # flake8: noqa: C901
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        image: Optional[np.ndarray] = None,
        channel_names: Optional[list[str]] = None,
        scale: Optional[float] = None,
        px_dimensions: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
    ):
        super().__init__(backing)
        self._available_resolutions = None
        self._base_resolution = None
        self._channel_names = None
        self._scale = None
        self._pxdim = None
        self._rotation = None
        self._translation = None
        self._images = {}

        if self.isbacked:
            resolutions = []
            for img in self.backing.values():
                resolutions.append([img.shape[1], img.shape[0]])

            attrs = self.backing.attrs
            if len(resolutions) > 1 or "base_resolution" in attrs:
                self._base_resolution = _read_hdf5_attribute(attrs, "base_resolution")
            elif len(resolutions) > 0:
                self._base_resolution = resolutions[0]

            self._available_resolutions = np.asarray(resolutions)
            self._channel_names = _get_hdf5_attribute(self.backing.attrs, "channel_names")
            self._scale = _get_hdf5_attribute(attrs, "scale")
            self._pxdim = _get_hdf5_attribute(attrs, "px_dimensions")
            self._rotation = _get_hdf5_attribute(attrs, "rotation")
            self._translation = _get_hdf5_attribute(attrs, "translation")

        if image is not None:
            if (
                self._available_resolutions is not None
                and len(self._available_resolutions) > 0
                and image is not None
            ):
                raise ValueError("trying to set image on a non-empty backing store")
            if (
                image.ndim == 3
                and image.shape[2] != 1
                and image.shape[2] != 3
                and channel_names is None
            ):
                raise ValueError("channel names are mandatory for images with 2 or > 3 channels")
            if channel_names is not None and (
                image.ndim == 3 and len(channel_names) != image.shape[3] or len(channel_names) != 1
            ):
                raise ValueError(
                    "the number of channel names must equal the number of channels in the image"
                )
            if px_dimensions is not None:
                px_dimensions = np.asarray(px_dimensions).squeeze()
                if px_dimensions.shape != (2,):
                    raise ValueError("px_dimensions must have shape (2,)")
            if rotation is not None:
                rotation = np.asarray(rotation).squeeze()
            if translation is not None:
                translation = np.asarray(translation).squeeze()

            self._base_resolution = np.asarray([image.shape[1], image.shape[0]])
            self._available_resolutions = self._base_resolution[np.newaxis, :]
            self._channel_names = channel_names
            self._scale = scale
            self._pxdim = px_dimensions
            self._rotation = rotation
            self._translation = translation
            if self.isbacked:
                self._write(self.backing)
                self._write_attributes(self.backing)
            else:
                self._images[(image.shape[1], image.shape[0])] = image

    @property
    def resolutions(self) -> Union[np.ndarray, None]:
        return self._available_resolutions

    @property
    def channel_names(self) -> Union[List[str], np.ndarray, None]:
        return self._channel_names

    def get_image(
        self, width: Optional[int] = None, downsample: bool = False
    ) -> Union[h5py.Group, np.ndarray]:
        # only width because we assume that all rescaling is proportional
        if width is None:
            width = self._base_resolution[0]
        idx = np.where(self._available_resolutions[:, 0] == width)[0]
        if idx.size == 0:
            if not downsample:
                raise IndexError(f"no image with width {width}")
            else:
                pass  # TODO
        else:
            if self.isbacked:
                res = self._available_resolutions[idx[0], :]
                return self.backing[f"{res[0]}x{res[1]}"][:]
            else:
                pass  # TODO

    def _write_attributes_impl(self, obj: h5py.Group):
        attrs = obj.attrs
        attrs["base_resolution"] = self._base_resolution
        if self._channel_names is not None:
            attrs["channel_names"] = self._channel_names
        if self._scale is not None:
            attrs["scale"] = self._scale
        if self._pxdim is not None:
            attrs["px_dimensions"] = self._pxdim
        if self._rotation is not None:
            attrs["rotation"] = self._rotation
        if self._translation is not None:
            attrs["translation"] = self._translation

    def _write(self, obj: h5py.Group):
        for res, img in self._images.items():
            dsetname = f"{res[0]}x{res[1]}"
            if dsetname in obj:
                del obj[dsetname]
            obj.create_dataset(dsetname, data=img, compression="gzip", compression_opts=9)

    def _set_backing(self, obj: Optional[h5py.Group]):
        if obj is not None:
            self._write(obj)
        elif self.isbacked:
            for img in self.backing:
                self._images[(img.shape[1], img.shape[0])] = img[:]

    @staticmethod
    def _encodingtype():
        return "image"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def rotation(self, resolution: Optional[tuple[int, int]] = None):
        return self._rotation

    def translation(self, resolution: Optional[tuple[int, int]] = None):
        return self._translation

    def pixel_dimensions(self, resolution: Optional[tuple[int, int]] = None):
        return self._pxdim

    @property
    def channel_names(self):
        return self._channel_names
