from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
import spatialmuon
from xml.etree import ElementTree
from scipy.sparse import csr_matrix
import anndata
import os
import h5py
import json
from PIL import Image


class Converter:
    def raster_from_tiff(self, path) -> spatialmuon.datatypes.raster.Raster:
        """Opens a .tiff and converts it to a spm.Raster()."""

        ome = tifffile.TiffFile(path)
        metadata = ElementTree.fromstring(ome.ome_metadata)[0]
        for chld in metadata:
            if chld.tag.endswith("Pixels"):
                metadata = chld
                break
        channel_names = []
        for channel in metadata:
            if channel.tag.endswith("Channel"):
                channel_names.append(channel.attrib["Fluor"])
        var = pd.DataFrame({"channel_name": channel_names})
        res = spatialmuon.Raster(X=np.moveaxis(ome.asarray(), 0, -1), var=var)

        return res

    def rastermask_from_tiff(self, path) -> spatialmuon._core.masks.RasterMasks:
        """Opens a .tiff and converts it to a spm.RasterMask()."""

        masks = np.asarray(tifffile.imread(path))
        res = spatialmuon.RasterMasks(X=masks)

        return res

    def regions_to_anndata(self, regions: spatialmuon.datatypes.regions.Regions) -> anndata.AnnData:
        var = regions.var.set_index("channel_name")
        adata = anndata.AnnData(X=regions.X, var=var)
        coords = regions.masks.obs[["region_center_y", "region_center_x"]].to_numpy()
        adata.obsm["spatial"] = coords
        adata.var.index = regions.var
        adata.var.reset_index(drop=True, inplace=True)
        return adata

    def read_visium10x(self, path: str):
        assert os.path.isdir(path)
        # visium_dir = os.path.join(path, 'outs')
        visium_dir = path
        count_matrix_file = os.path.join(visium_dir, "filtered_feature_bc_matrix.h5")
        image_file = os.path.join(visium_dir, "spatial/tissue_hires_image.png")
        coords_file = os.path.join(visium_dir, "spatial/tissue_positions_list.csv")
        scale_factors_file = os.path.join(visium_dir, "spatial/scalefactors_json.json")
        for f in [count_matrix_file, image_file, coords_file, scale_factors_file]:
            assert os.path.isfile(f)

        modality = spatialmuon.SpatialModality()

        with h5py.File(os.path.join(count_matrix_file)) as f:
            matrix = f["matrix"]
            X = csr_matrix(
                (matrix["data"][()], matrix["indices"][()], matrix["indptr"][()]),
                shape=matrix["shape"][()][::-1],
            )

            barcodes = matrix["barcodes"].asstr()[()]

            var = pd.DataFrame(dict(channel_name=matrix["features/name"].asstr()[()]))
            var["id"] = matrix["features/id"].asstr()[()]
            for fname in matrix["features/_all_tag_keys"].asstr()[()]:
                feat = matrix[f"features/{fname}"]
                if h5py.check_string_dtype(feat.dtype):
                    feat = feat.asstr()
                var[fname] = feat[()]

        tissue_positions = (
            pd.read_csv(
                os.path.join(coords_file),
                names=(
                    "barcode",
                    "in_tissue",
                    "array_row",
                    "array_col",
                    "pxl_col_in_fullres",
                    "pxl_row_in_fullres",
                ),
            )
            .set_index("barcode")
            .loc[barcodes]
            .drop("in_tissue", axis=1)
        )
        coords = tissue_positions[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
        obs = tissue_positions.drop(["pxl_row_in_fullres", "pxl_col_in_fullres"], axis=1)

        with open(os.path.join(scale_factors_file), "r") as f:
            meta = json.load(f)

        # center_to_center = meta['spot_diameter_fullres'] / 55 * 100 * meta['tissue_hires_scalef']
        # radius = center_to_center * 55 / 100 / 2
        radius = meta["spot_diameter_fullres"] * meta["tissue_hires_scalef"] / 2
        coords = coords * meta["tissue_hires_scalef"]
        anchor = spatialmuon.Anchor(vector=np.array([radius / (55 / 2), 0.0]))

        # the samples are offset by 10 μm in the Z axis according to the paper
        # I have no idea how much that is in pixels
        # So just do 10 px
        labels = obs.index.tolist()
        masks = spatialmuon.ShapeMasks(
            masks_shape="circle", masks_centers=coords, masks_radii=radius, masks_labels=labels
        )
        # scale = 6.698431978755106
        cfov = spatialmuon.datatypes.regions.Regions(
            X=X, var=var, masks=masks, anchor=anchor, coordinate_unit="um"
        )
        modality["expression"] = cfov

        img = Image.open(os.path.join(image_file))
        hires_img = np.asarray(img)
        img.close()
        if np.min(hires_img) < 0 or np.max(hires_img) > 1:
            assert hires_img.dtype == np.dtype("uint8")
            hires_img = hires_img / 255
        modality[f"image"] = spatialmuon.datatypes.raster.Raster(
            X=hires_img, anchor=anchor, coordinate_unit="um"
        )
        return modality

        # outfname = os.path.join(path, 'visium.h5smu')
        # if os.path.isfile(outfname):
        #     os.unlink(outfname)

        # smudata = spatialmuon.SpatialMuData(outfname, backingmode="w")
        # smudata = spatialmuon.SpatialModality()
        # smudata["visium"] = modality
        # return smudata


if __name__ == "__main__":
    f = "/Users/macbook/temp"
    modality = Converter().read_visium10x(f)
    smu = spatialmuon.SpatialMuData("/Users/macbook/temp/test.h5smu", backingmode="w")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    smu["visium"] = modality
    smu["visium"]["image"].plot(ax=ax, alpha=0.4)
    smu["visium"]["expression"].plot(0, ax=ax)
    plt.show()
