from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import spatialmuon
import math


class Tiles:
    def __init__(
        self,
        raster: "Raster",
        masks: "Masks",
        tile_dim: int,
    ):
        self.raster = raster
        self.masks = masks
        self.tile_dim = tile_dim
        raster_tiles, masks_tiles, self.centers, self.origins = masks.extract_tiles(
            raster, tile_dim=tile_dim
        )
        self.raster_tiles = np.stack(raster_tiles)
        self.masks_tiles = np.stack(masks_tiles)
        assert self.raster_tiles.shape[:3] == self.masks_tiles.shape
        if True:
            ##
            n = 2
            c = 0
            _, ax = plt.subplots(1)
            s = self.mask_tile_to_raster(index=n)
            # s = self.raster_tile_to_raster(index=n)
            s.plot(channels=c, ax=ax)
            # self.raster.plot(channels=c, ax=ax)
            self.masks.plot(fill_colors="red", outline_colors="k", ax=ax, alpha=0.5)
            plt.show()
            ##

    # TODO: tiles plotted back are not pixel perfect, fix it
    def get_anchor_for_tile(self, index: int):
        origin = self.origins[index]
        anchor = spatialmuon.Anchor(origin=origin)
        return anchor

    def raster_tile_to_raster(self, index: int):
        tile = self.raster_tiles[index]
        anchor = self.get_anchor_for_tile(index)
        s = spatialmuon.Raster(X=tile, anchor=anchor)
        return s

    def mask_tile_to_raster_mask(self, index: int):
        raise NotImplementedError("masks do not support anchors yet")
        # tile = self.masks_tiles[index]
        # anchor = self.get_anchor_for_tile(index)
        # s = spatialmuon.RasterMasks(X)

    def mask_tile_to_raster(self, index: int):
        tile = self.masks_tiles[index]
        tile = tile[..., np.newaxis]
        anchor = self.get_anchor_for_tile(index)
        s = spatialmuon.Raster(X=tile, anchor=anchor)
        return s
