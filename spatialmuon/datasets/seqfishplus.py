#!/usr/bin/env python3

# SeqFISH+ data from Eng et al. (2019), doi:10.1038/s41586-019-1049-y

import tempfile
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.io import loadmat
from PIL import Image
import roifile

import spatialmuon
from spatialmuon.datasets._utils import download, unzip

if len(sys.argv) > 1:
    outfname = sys.argv[1]
else:
    outfname = "seqfishplus.h5smu"

urls = {
    "point_locations": "https://zenodo.org/record/2669683/files/seqFISH%2B_NIH3T3_point_locations.zip?download=1",
    "images1": "https://zenodo.org/record/2669683/files/DAPI_experiment1.zip?download=1",
    "images2": "https://zenodo.org/record/2669683/files/DAPI_experiment2.zip?download=1",
    "rois1": "https://zenodo.org/record/2669683/files/ROIs_Experiment1_NIH3T3.zip?download=1",
    "rois2": "https://zenodo.org/record/2669683/files/ROIs_Experiment2_NIH3T3.zip?download=1",
}


downloaded = {
    "point_locations": "point_locations.zip",
    "images1": "images1.zip",
    "images2": "images2.zip",
    "rois1": "rois1.zip",
    "rois2": "rois2.zip",
}

DEBUG = False
# DEBUG = True

DOWNLOAD = True
if DEBUG:
    DOWNLOAD = False

with tempfile.TemporaryDirectory() as tmpdir:
    if not DOWNLOAD:
        download_dir = "/data/spatialmuon/datasets/seqfishplus/raw/"
    else:
        download_dir = tmpdir

    for k, v in downloaded.items():
        downloaded[k] = os.path.join(download_dir, v)

    if not DEBUG:
        for k, v in downloaded.items():
            dest = v
            url = urls[k]
            download(url, dest, desc=f"downloading {k}")

    for k, v in downloaded.items():
        unzip(v, tmpdir, rm=not DEBUG)
    os.rename(os.path.join(tmpdir, "ALL_Roi"), os.path.join(tmpdir, "rois1"))
    os.rename(os.path.join(tmpdir, "ROIs"), os.path.join(tmpdir, "rois2"))

    if os.path.isfile(outfname):
        os.unlink(outfname)
    smudata = spatialmuon.SpatialMuData(outfname)
    modality = spatialmuon.SpatialModality(coordinate_unit="px")
    smudata["SeqFISH+"] = modality

    gene_names = [
        str(g)
        for g in np.concatenate(
            loadmat(os.path.join(tmpdir, "all_gene_Names.mat"))["allNames"].squeeze()
        )
    ]

    for run in (1, 2):
        imgdir = os.path.join(tmpdir, f"final_background_experiment{run}")
        positions = loadmat(os.path.join(tmpdir, f"RNA_locations_run_{run}.mat"))["tot"]
        nfov, ncell, ngene = positions.shape
        for fov in range(nfov):
            if DEBUG:
                if fov > 0:
                    break
            cellids = []
            coords = []
            nspots = 0
            feature_name = []
            for gene in range(ngene):
                if DEBUG:
                    if gene > 1:
                        break
                cnspots = 0
                for cell in range(ncell):
                    ccoords = positions[fov, cell, gene]
                    if ccoords.size > 0:
                        cellids.extend([cell] * ccoords.shape[0])
                        coords.append(ccoords[:, :2])
                        cnspots += ccoords.shape[0]
                nspots += cnspots
                feature_name.extend([gene_names[gene]] * cnspots)

            coords = np.concatenate(coords, axis=0)
            coords = gpd.GeoDataFrame(
                {"cell": cellids}, index=feature_name, geometry=[Point(*c) for c in coords]
            )

            img = Image.open(os.path.join(imgdir, f"MMStack_Pos{fov}.ome.tif"))
            img.seek(7)
            dapi_img = np.asarray(img)
            img.close()
            translation = [
                fov * (dapi_img.shape[0] + np.floor(0.05 * dapi_img.shape[0])),
                run * (dapi_img.shape[1] + np.floor(0.05 * dapi_img.shape[0])),
            ]

            cfov = spatialmuon.SingleMolecule(
                data=coords,
                index_kwargs={
                    "progressbar": True,
                    "desc": f"creating spatial index for run {run} FOV {fov}",
                },
                translation=translation,
            )
            modality[f"run{run}_fov{fov}"] = cfov
            modality[f"run{run}_fov{fov}_dapi"] = spatialmuon.Raster(X=dapi_img)

            masks = spatialmuon.PolygonMasks()
            regions = spatialmuon.Regions(masks=masks)
            modality[f"run{run}_fov{fov}_rois"] = regions
            roidir = os.path.join(tmpdir, f"rois{run}")
            with os.scandir(os.path.join(roidir, f"RoiSet_Pos{fov}")) as rdir:
                for rfile in rdir:
                    roi = roifile.roiread(rfile.path)
                    masks[roi.name] = roi.coordinates()
