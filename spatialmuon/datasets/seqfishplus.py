#!/usr/bin/env python3

# SeqFISH+ data from Eng et al. (2019), doi:10.1038/s41586-019-1049-y

import tempfile
import urllib.request
import zipfile
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.io import loadmat
from PIL import Image
import roifile

from tqdm import tqdm
import h5py

import spatialmuon
from spatialmuon.datatypes.singlemolecule import SingleMolecule

if len(sys.argv) > 1:
    outfname = sys.argv[1]
else:
    outfname = "seqfishplus.h5smu"

class TqdmDownload(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.update({"unit": "B", "unit_scale": True, "unit_divisor": 1024})
        super().__init__(*args, **kwargs)

    def update_to(self, nblocks=1, blocksize=1, total=-1):
        self.total = total
        self.update(nblocks * blocksize - self.n)

def download(url, outfile, desc):
    with TqdmDownload(desc="downloading " + desc) as t:
        urllib.request.urlretrieve(url, outfile, t.update_to)

def unzip(file, outdir):
    zfile = zipfile.ZipFile(file)
    os.makedirs(outdir, exist_ok=True)
    zfile.extractall(outdir)
    zfile.close()

with tempfile.TemporaryDirectory() as tmpdir:
    if os.path.isfile(outfname):
        os.unlink(outfname)
    smudata = spatialmuon.SpatialMuData(outfname)
    modality = spatialmuon.SpatialModality(coordinate_unit="px")
    smudata["SeqFISH+"] = modality

    locationsfile = os.path.join(tmpdir, "point_locations.zip")
    download("https://zenodo.org/record/2669683/files/seqFISH%2B_NIH3T3_point_locations.zip?download=1", locationsfile, desc="point locations")
    locationsdir = os.path.join(tmpdir, "point_locations")
    unzip(locationsfile, locationsdir)

    gene_names = [str(g) for g in np.concatenate(loadmat(os.path.join(locationsdir, "all_gene_Names.mat"))["allNames"].squeeze())]

    for run, imgurl, roiurl in zip(
        (1, 2),
        ("https://zenodo.org/record/2669683/files/DAPI_experiment1.zip?download=1",
         "https://zenodo.org/record/2669683/files/DAPI_experiment2.zip?download=1"),
        ("https://zenodo.org/record/2669683/files/ROIs_Experiment1_NIH3T3.zip?download=1",
         "https://zenodo.org/record/2669683/files/ROIs_Experiment2_NIH3T3.zip?download=1"),
        ):
        imgzipfile = os.path.join(tmpdir, "images.zip")
        download(imgurl, imgzipfile, desc=f"run {run} images")
        unzip(imgzipfile, tmpdir)
        imgdir = os.path.join(tmpdir, f"final_background_experiment{run}")

        roizipfile = os.path.join(tmpdir, "rois.zip")
        roidir = os.path.join(tmpdir, f"rois_run{run}")
        download(roiurl, roizipfile, desc=f"run {run} ROIs")
        unzip(roizipfile, roidir)

        positions = loadmat(os.path.join(locationsdir, f"RNA_locations_run_{run}.mat"))["tot"]
        nfov, ncell, ngene = positions.shape
        for fov in range(nfov):
            cellids = []
            coords = []
            nspots = 0
            feature_name = []
            for gene in range(ngene):
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
            coords = gpd.GeoDataFrame({"cell": cellids}, index=feature_name, geometry=[Point(*c) for c in coords])

            img = Image.open(os.path.join(imgdir, f"MMStack_Pos{fov}.ome.tif"))
            img.seek(7)
            dapi_img = np.asarray(img)
            img.close()
            translation = [fov * (dapi_img.shape[0] + np.floor(0.05 * dapi_img.shape[0])), run * (dapi_img.shape[1] + np.floor(0.05 * dapi_img.shape[0]))]

            cfov = SingleMolecule(data=coords, index_kwargs={"progressbar":True, "desc": f"creating spatial index for run {run} FOV {fov}"}, translation=translation)
            modality[f"run{run}_fov{fov}"] = cfov
            cfov.images["DAPI"] = spatialmuon.Image(image=dapi_img)

            mask = spatialmuon.PolygonMask()
            cfov.feature_masks["ROIs"] = mask
            with os.scandir(os.path.join(roidir, os.listdir(roidir)[0], f"RoiSet_Pos{fov}")) as rdir:
                for rfile in rdir:
                    roi = roifile.roiread(rfile.path)
                    mask[roi.name] = roi.coordinates()

