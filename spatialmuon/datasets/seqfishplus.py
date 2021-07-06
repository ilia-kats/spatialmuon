#!/usr/bin/env python3

# SeqFISH+ data from Eng et al. (2019), doi:10.1038/s41586-019-1049-y

import tempfile
import urllib.request
import zipfile
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
from PIL import Image
import roifile

from tqdm import tqdm
import h5py
import anndata as ad

from spatialmuon import SerializableStorage
from rtree import index

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

with tempfile.TemporaryDirectory() as tmpdir, h5py.File(outfname, "w", userblock_size=512, libver="latest") as outfile:
    outfile.attrs["encoder"] = "seqfishplus-downloader"
    outfile.attrs["encoder-version"] = "0.1.0"
    outfile.attrs["encoding"] = "SpatialMuData"
    outfile.attrs["encoding-version"] = "0.1.0"
    modality = outfile.create_group("/mod/SeqFISH+")
    modality.attrs["encoding"] = "spatialmodality"
    modality.attrs["encoding-version"] = "0.1.0"
    modality.attrs["coordinate_unit"] = "px"

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
            fovgrp = modality.create_group(f"run{run}_fov{fov}")
            fovgrp.attrs["encoding"] = "single-molecule"
            fovgrp.attrs["encoding-version"] = "0.1.0"

            feature_range = fovgrp.create_group("feature_range")

            storage = SerializableStorage()
            p = index.Property(type=index.RT_RTree, variant=index.RT_Star, dimension=2)
            idx = index.Index(storage, interleaved=True, properties=p)

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
                feature_range[gene_names[gene]] = [nspots, nspots + cnspots]
                nspots += cnspots
                feature_name.extend([gene_names[gene]] * cnspots)

            coords = np.concatenate(coords, axis=0)
            cellids = pd.DataFrame({"cell": cellids}, index=feature_name)
            fovgrp.create_dataset("coordinates", data=coords, compression="gzip", compression_opts=9)
            ad._io.h5ad.write_attribute(fovgrp, "metadata", cellids, dataset_kwargs={"compression": "gzip", "compression_opts":9})

            for i, c in enumerate(tqdm(coords, desc=f"creating spatial index for run {run} FOV {fov}")):
                idx.insert(i, np.hstack((c, c)))
            storage.to_hdf5(fovgrp, "index")

            img = Image.open(os.path.join(imgdir, f"MMStack_Pos{fov}.ome.tif"))
            img.seek(7)
            dapi_img = np.asarray(img)
            img.close()
            img_grp = fovgrp.create_group(f"images/{dapi_img.shape[1]}x{dapi_img.shape[0]}")
            img_grp.create_dataset("image", data=dapi_img, compression="gzip", compression_opts=9)
            fovgrp["translation"] = [fov * dapi_img.shape[0] + np.floor(0.05 * dapi_img.shape[0]).astype(np.int16), 0]

            maskgrp = fovgrp.create_group("feature_masks/ROIs")
            maskgrp.attrs["encoding"] = "polygon"
            maskgrp.attrs["encoding-version"] = "0.1.0"
            with os.scandir(os.path.join(roidir, os.listdir(roidir)[0], f"RoiSet_Pos{fov}")) as rdir:
                for rfile in rdir:
                    roi = roifile.roiread(rfile.path)
                    maskgrp.create_dataset(roi.name, data=roi.coordinates(), compression="gzip", compression_opts=9)


with open(outfname, "rb+") as outfile:
    outfile.write(b"SpatialMuData (format-version=0.1.0;creator=seqfishplus-downloader;creator-version=0.1.0)")
