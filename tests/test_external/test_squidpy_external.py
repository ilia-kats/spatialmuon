##
import unittest
import os
import sys
import spatialmuon
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import copy
from pathlib import Path
from spatialmuon.external.squidpy_external import SquidpyExternal
import scanpy
import warnings

DEBUGGING = False
try:
    __file__
except NameError as e:
    if str(e) == "name '__file__' is not defined":
        DEBUGGING = True
    else:
        raise e
if sys.gettrace() is not None:
    DEBUGGING = True

if not DEBUGGING:
    # Get current file and pre-generate paths and names
    this_dir = Path(__file__).parent
    fpath = this_dir / "../data/small_imc.h5smu"

    matplotlib.use("Agg")
else:
    fpath = os.path.expanduser("~/spatialmuon/tests/data/small_imc.h5smu")

plt.style.use("dark_background")


##
class SquidpyExternal_TestClass(unittest.TestCase):
    def test_can_get_squidpy_data_representation(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        accumulated = d["imc"]["ome"].accumulate_features(d["imc"]["masks"].masks)
        e = accumulated["mean"]
        se = SquidpyExternal()
        adata = se.get_data_rerepsentation(e, raster_images=d["imc"]["ome"])
        return adata

    def test_can_compute_spatial_neighbors(self):
        adata = self.test_can_get_squidpy_data_representation()
        se = SquidpyExternal()
        se.compute_spatial_neighbors(adata)

    def get_louvain_clustered_data(self):
        adata = self.test_can_get_squidpy_data_representation()
        adata.X = np.arcsinh(adata.X)
        scanpy.pp.neighbors(adata)
        # the higher the resolution, the more the clusters
        scanpy.tl.louvain(adata, resolution=1.1)
        scanpy.tl.umap(adata)
        return adata

    def test_cluster_with_scanpy_and_plot_with_spatialmuon(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        accumulated = d["imc"]["ome"].accumulate_features(d["imc"]["masks"].masks)
        e = accumulated["mean"]
        se = SquidpyExternal()
        adata = se.get_data_rerepsentation(e, raster_images=d["imc"]["ome"])
        adata.X = np.arcsinh(adata.X)
        scanpy.pp.neighbors(adata)
        # the higher the resolution, the more the clusters
        scanpy.tl.louvain(adata, resolution=1.1)
        louvain = adata.obs["louvain"]
        masks = copy.copy(e.masks)
        louvain.index = masks.obs.index
        masks.obs["louvain"] = louvain
        clustered = spatialmuon.Regions(backing=None, X=None, index_kwargs={}, masks=masks)
        fig, ax = plt.subplots(1, figsize=(5, 5))
        d["imc"]["ome"].plot(
            channels=0, ax=ax, cmap=matplotlib.cm.get_cmap("Greys_r"), show_title=False
        )
        clustered.masks.plot(
            fill_colors=None, outline_colors="louvain", background_color=(0.0, 0.0, 0.0, 0.5), ax=ax
        )
        plt.tight_layout()
        plt.show()

    def test_neighbors_enrichment_analysis(self):
        adata = self.get_louvain_clustered_data()
        scanpy.pl.umap(adata, color="louvain")
        se = SquidpyExternal()
        se.compute_spatial_neighbors(adata)
        se.neighbors_enrichment_analysis(adata, cluster_key="louvain")

    def test_view_with_napari(self):
        self.skipTest(
            "skipping the integration with Napari as the machine used for testing has no remote display "
            "set up"
        )
        # adata = self.get_louvain_clustered_data()
        # se = SquidpyExternal()
        # se.view_with_napari(adata)


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main()
    else:
        # SquidpyExternal_TestClass().test_can_get_squidpy_data_representation()
        # SquidpyExternal_TestClass().test_can_compute_spatial_neighbors()
        SquidpyExternal_TestClass().test_cluster_with_scanpy_and_plot_with_spatialmuon()
        # SquidpyExternal_TestClass().test_neighbors_enrichment_analysis()
        # SquidpyExternal_TestClass().test_view_with_napari()
