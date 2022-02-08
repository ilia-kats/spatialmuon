from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import spatialmuon
import math
from functools import singledispatchmethod
from typing import Union, Optional
from spatialmuon.datatypes.raster import Raster
from spatialmuon._core.masks import Masks, RasterMasks, ShapeMasks
from scipy.ndimage import center_of_mass

DEBUG_WITH_PLOTS = False


class Tiles:
    def __init__(
        self,
        masks: Masks,
        raster: Optional[Raster],
        tile_dim_in_units: Optional[float] = None,
        tile_dim_in_pixels: Optional[float] = None,
    ):
        self.raster = raster
        self.masks = masks

        assert (tile_dim_in_units is None) != (tile_dim_in_pixels is None)
        if tile_dim_in_units is None:
            if not np.isclose(tile_dim_in_pixels, int(tile_dim_in_pixels)):
                raise ValueError("tile_dim_in_pixels need to represent an integer value")
            self.tile_dim_in_pixels = tile_dim_in_pixels
            self.tile_dim_in_units = self.target.anchor.transform_length(self.tile_dim_in_pixels)
        else:
            self.tile_dim_in_pixels = round(
                self.target.anchor.inverse_transform_length(tile_dim_in_units)
            )
            self.tile_dim_in_units = self.target.anchor.transform_length(self.tile_dim_in_pixels)

        if self.raster is not None:
            self.tiles, self.origins = self._extract_tiles(
                self.masks, self.raster, tile_dim_in_units=self.tile_dim_in_units
            )
        else:
            self.tiles, self.origins = self._isolate_masks_into_tiles(
                self.masks, tile_dim_in_units=self.tile_dim_in_units
            )
        if True:
            ##
            n = 0
            _, ax = plt.subplots(1)
            s = self.tile_to_raster(index=n)
            c = list(range(min(3, len(s.var))))
            # s = self.raster_tile_to_raster(index=n)
            s.plot(channels=c, ax=ax)
            # self.raster.plot(channels=c, ax=ax)
            self.masks.plot(fill_colors="red", outline_colors="k", ax=ax, alpha=0.2)
            plt.show()
            ##

    @property
    def target(self):
        if self.raster is not None:
            return self.raster
        else:
            return self.masks

    def get_anchor_for_tile(self, index: int):
        origin = self.origins[index]
        anchor = spatialmuon.Anchor(origin=origin)
        composed = spatialmuon.Anchor.compose_anchors(anchor, self.target.anchor)
        return composed

    def tile_to_raster(self, index: int):
        tile = self.tiles[index]
        if isinstance(self.raster, spatialmuon.RasterMasks) and len(tile.shape) == 2:
            tile = tile[..., np.newaxis]
        anchor = self.get_anchor_for_tile(index)
        # notice that tiles from raster masks are also converted to raster; we will add a small helper function to
        # get raster masks from a raster object
        s = spatialmuon.Raster(X=tile, anchor=anchor, coordinate_unit=self.target.coordinate_unit)
        return s

    def _extract_tiles(self, masks: Masks, raster: Raster, tile_dim_in_units: float):
        global DEBUG_WITH_PLOTS

        extracted_tiles = []
        origins = []
        tile_dim_in_pixels = round(raster.anchor.inverse_transform_length(tile_dim_in_units))
        transformed_centers = masks.transformed_masks_centers
        raster_centers = raster.anchor.inverse_transform_coordinates(transformed_centers)

        for i, xy in enumerate(raster_centers):
            tile, origin = self._extract_tile_around_point(
                tensor=raster.X, center_in_pixel=xy, tile_dim_in_pixels=tile_dim_in_pixels
            )
            extracted_tiles.append(tile)
            origins.append(origin)
            if DEBUG_WITH_PLOTS:
                if i >= 4:
                    DEBUG_WITH_PLOTS = False
        return extracted_tiles, origins

    @singledispatchmethod
    def _isolate_masks_into_tiles(self, masks: Masks, tile_dim_in_units: float):
        raise NotImplementedError()

    @_isolate_masks_into_tiles.register
    def _(self, masks: RasterMasks, tile_dim_in_units: float):
        global DEBUG_WITH_PLOTS

        mask_labels = set(masks.obs["original_labels"].to_list())

        # integrity check
        real_labels = set(np.unique(masks.X).tolist())
        if 0 in real_labels:
            real_labels.remove(0)
        assert mask_labels == real_labels

        extracted_masks = []
        origins = []
        tile_dim_in_pixels = round(masks.anchor.inverse_transform_length(tile_dim_in_units))
        masks = masks.X[...]

        for i, mask_label in tqdm(
            zip(range(len(mask_labels)), mask_labels),
            desc="extracting tiles",
            total=len(mask_labels),
        ):
            z = masks == mask_label
            z_center = center_of_mass(z)
            xy = np.array((z_center[1], z_center[0]))
            z = z[..., np.newaxis]
            tile, origin = self._extract_tile_around_point(
                tensor=z, center_in_pixel=xy, tile_dim_in_pixels=tile_dim_in_pixels
            )

            extracted_masks.append(tile)
            origins.append(origin)
            if DEBUG_WITH_PLOTS:
                if i >= 4:
                    DEBUG_WITH_PLOTS = False
        return extracted_masks, origins

    @staticmethod
    def _extract_tile_around_point(tensor, center_in_pixel: np.ndarray, tile_dim_in_pixels: float):
        # large image
        LAZY_LOADING = np.prod(tensor.shape) > 5000 * 5000 * 3
        if LAZY_LOADING:
            x = tensor
        else:
            x = tensor[...]
        t = tile_dim_in_pixels
        r = math.floor(t / 2)

        a0, b0, a1, b1, origin_x, origin_y = Tiles._get_coords_for_cropping(
            mask_center=center_in_pixel, r=r, masks_shape=x.shape
        )
        origin = np.array([origin_x, origin_y])
        y = x[a1:b1, a0:b0, :]
        y_center = np.array([center_in_pixel[0] - a0, center_in_pixel[1] - a1])

        src0_a, src0_b, des0_a, des0_b = Tiles._get_coords_for_padding(
            r, y.shape[1], int(y_center[0])
        )
        src1_a, src1_b, des1_a, des1_b = Tiles._get_coords_for_padding(
            r, y.shape[0], int(y_center[1])
        )

        tile = np.zeros((t, t, x.shape[2]))
        tile[des1_a:des1_b, des0_a:des0_b, :] = y[src1_a:src1_b, src0_a:src0_b, :]

        if DEBUG_WITH_PLOTS:
            ##
            plt.figure(figsize=(10, 3))
            plt.suptitle("visualizing the tile extraction")
            plt.subplot(1, 3, 1)
            if len(x.shape) == 3:
                x_viz = x[..., 0]
            plt.imshow(x_viz, origin="lower", extent=(0, x_viz.shape[1], 0, x_viz.shape[0]))
            plt.axvline(x=a0, lw="3")
            plt.axvline(x=b0, lw="3")
            plt.axhline(y=a1, lw="3")
            plt.axhline(y=b1, lw="3")
            plt.title(f"x: {a0}:{b0}, y: {a1}:{b1}")

            plt.subplot(1, 3, 2)
            if len(y.shape) == 3:
                y_viz = y[..., 0]
            plt.imshow(y_viz, origin="lower", extent=(0, y_viz.shape[1], 0, y_viz.shape[0]))
            plt.scatter(y_center[0], y_center[1], s=5, c="r")

            plt.subplot(1, 3, 3)
            if len(tile.shape) == 3:
                tile_viz = tile[..., 0]
            plt.imshow(tile_viz, origin="lower", extent=(0, tile_viz.shape[1], 0, tile_viz.shape[0]))
            plt.scatter(r, r, color="red", s=1)
            plt.title(
                f"x: {src1_a}:{src1_b} -> {des1_a}:{des1_b}, y: {src0_a}:{src0_b} -> {des0_a}:{des0_b}"
            )
            plt.show()
            ##

        return tile, origin

    @staticmethod
    def _get_coords_for_cropping(mask_center, r, masks_shape):
        a0 = math.floor(mask_center[0] - r)
        origin_x = a0
        a0 = max(0, a0)
        b0 = math.ceil(mask_center[0] + r)
        b0 = min(b0, masks_shape[1])

        a1 = math.floor(mask_center[1] - r)
        origin_y = a1
        a1 = max(0, a1)
        b1 = math.ceil(mask_center[1] + r)
        b1 = min(b1, masks_shape[0])
        return a0, b0, a1, b1, origin_x, origin_y

    @staticmethod
    def _get_coords_for_padding(des_r, src_l, src_c):
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
