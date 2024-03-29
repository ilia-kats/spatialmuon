#!/usr/bin/env python3

# Visium mouse brain data from Kleshchevnikov et al., 2020 (doi:10.1101/2020.11.15.378125)

import tempfile
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import geopandas as gpd
from shapely.geometry import Point
from PIL import Image
from spatialmuon._core.masks import ShapeMasks

from tqdm import tqdm
import h5py

import spatialmuon
from spatialmuon.datasets._utils import download, unzip, md5


# DEBUG = False
DEBUG = True

DOWNLOAD = False
if DEBUG:
    DOWNLOAD = False

if len(sys.argv) > 1:
    outfname = sys.argv[1]
else:
    outfname = "/data/spatialmuon/datasets/visium_mousebrain/smu/visium.h5smu"

with tempfile.TemporaryDirectory() as tmpdir:
    if not DOWNLOAD:
        download_dir = "/data/spatialmuon/datasets/visium_mousebrain/raw/"
    else:
        download_dir = tmpdir
    brainfile = os.path.join(download_dir, "mouse_brain.zip")
    if DOWNLOAD:
        download(
            "https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_visium_wo_cloupe_data.zip",
            brainfile,
            desc="data",
        )
    unzip(brainfile, tmpdir, rm=DOWNLOAD)

    if os.path.isfile(outfname):
        os.unlink(outfname)

    smudata = spatialmuon.SpatialMuData(outfname, backingmode="w")
    smudata["Visium"] = modality = spatialmuon.SpatialModality()

    fovdir = os.path.join(tmpdir, "mouse_brain_visium_wo_cloupe_data", "rawdata")
    fovs = [f for f in os.listdir(fovdir) if not f.startswith(".")]
    with tqdm(total=len(fovs)) as pbar:
        for fovidx, fovname in enumerate(fovs):
            pbar.set_description(f"processing slide {fovname}")
            cdir = os.path.join(fovdir, fovname)
            with h5py.File(os.path.join(cdir, "filtered_feature_bc_matrix.h5")) as f:
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
                    os.path.join(cdir, "spatial", "tissue_positions_list.csv"),
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

            with open(os.path.join(cdir, "spatial", "scalefactors_json.json"), "r") as f:
                meta = json.load(f)
            # wrong
            # radius = 0.5 * meta["spot_diameter_fullres"] * meta["tissue_hires_scalef"]
            # computed manually by considering pairwise differences of the xaxis, using the following code and some
            # extra easy steps
            # import matplotlib.pyplot as plt
            # plt.scatter(coords[:, 0], coords[:, 1], s=1)
            # plt.gca().set_aspect('equal')
            # plt.gca().set(xlim=(1000, 1100), ylim=(800, 1000))
            # x = coords[:, 0]
            # x = x[(x > 1000) * (x < 1100)]
            # diff = (x[:, np.newaxis] - x[np.newaxis, :]).flatten()
            # plt.show()
            # plt.hist(x, bins=100)
            # plt.show()

            center_to_center = 25.411068101069596
            radius = center_to_center * 55 / 100 / 2
            coords = coords * meta["tissue_hires_scalef"]
            anchor = spatialmuon.Anchor(vector=np.array([center_to_center / 100, 0.0]))

            # the samples are offset by 10 μm in the Z axis according to the paper
            # I have no idea how much that is in pixels
            # So just do 10 px
            labels = obs.index.tolist()
            masks = ShapeMasks(
                masks_shape="circle", masks_centers=coords, masks_radii=radius, masks_labels=labels
            )
            # scale = 6.698431978755106
            cfov = spatialmuon.Regions(
                X=X, var=var, masks=masks, anchor=anchor, coordinate_unit="um"
            )
            modality[fovname] = cfov

            img = Image.open(os.path.join(cdir, "spatial", "tissue_hires_image.png"))
            hires_img = np.asarray(img)
            img.close()
            modality[f"{fovname}H&E"] = spatialmuon.Raster(
                X=hires_img, anchor=anchor, coordinate_unit="um"
            )
            # cfov.images["H&E"] = spatialmuon.Image(image=hires_img)

            pbar.update()
            if DEBUG:
                print("debugging: stopping at the first slide")
                break
