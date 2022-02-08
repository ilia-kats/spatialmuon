import anndata
import squidpy
import spatialmuon
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional, List


class SquidpyExternal:
    def get_data_rerepsentation(
        self,
        regions: spatialmuon.datatypes.regions.Regions,
        raster_images: Optional[
            Union[spatialmuon.datatypes.raster.Raster, List[spatialmuon.datatypes.raster.Raster]]
        ] = None,
    ) -> anndata.AnnData:
        c = spatialmuon.processing.converter.Converter()
        adata = c.regions_to_anndata(regions)
        if raster_images is not None:
            if isinstance(raster_images, spatialmuon.datatypes.raster.Raster):
                x = raster_images.X[...]
                assert len(x.shape) in [2, 3]
                if len(x.shape) == 2:
                    x = np.unsqueeze(x, 2)
                for i in range(x.shape[2]):
                    img = x[:, :, i]
                    # name = f"channel{i}"
                    name = f"{i}"
                    if "spatial" not in adata.uns:
                        adata.uns["spatial"] = dict()
                    adata.uns["spatial"][name] = dict()
                    adata.uns["spatial"][name]["images"] = {"hires": img}
                    # the handling of scalefactors is not implemented yet
                    adata.uns["spatial"][name]["scalefactors"] = {
                        "tissue_hires_scalef": 1.0,
                        "spot_diameter_fullres": 2.5,
                    }
            elif isinstance(raster_images, list) and all(
                [isinstance(r, spatialmuon.datatypes.raster.Raster) for r in raster_images]
            ):
                raise NotImplementedError()
            else:
                raise ValueError(
                    "raster_images can be either None, a Raster object, or a list of Raster"
                )

        return adata

    def compute_spatial_neighbors(self, adata: anndata.AnnData):
        squidpy.gr.spatial_neighbors(adata)
        return adata

    def neighbors_enrichment_analysis(self, adata: anndata.AnnData, cluster_key: str):
        squidpy.gr.nhood_enrichment(adata, cluster_key=cluster_key)
        n = len(adata.uns["louvain_colors"])
        squidpy.pl.nhood_enrichment(adata, cluster_key=cluster_key, figsize=(n, n))
        plt.tight_layout()
        plt.show()

    def view_with_napari(self, adata):
        i = 0
        # name = f"channel{i}"
        name = f"{3}"
        image = squidpy.im.ImageContainer(
            adata.uns["spatial"][name]["images"]["hires"],
            scale=adata.uns["spatial"][name]["scalefactors"]["tissue_hires_scalef"],
        )
        # the following lines returns immediately, so to see the napari window run this in an interactive console
        viewer = image.interactive(adata, library_id=[name])
        pass
