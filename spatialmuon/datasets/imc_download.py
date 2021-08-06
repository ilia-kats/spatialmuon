#!/usr/bin/env python3

# IMC data from Jackson et al. (2020), doi:10.1038/s41586-019-1876-x

import tempfile
import os
import sys
import math
from typing import List, Dict, Union, Optional
import re
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import tifffile

import spatialmuon
from _utils import *

DISEASE_STATUSES = ["non-tumor", "tumor"]
CANCER_SUBTYPES = ["PR+ER+", "PR-ER+", "PR-ER-", "PR+ER-"]
CANCER_CLINICAL_TYPES = ["HR+HER2-", "TripleNeg", "HR+HER2+", "HR-HER2+"]
# using the labels proposed here: https://www.cancerresearchuk.org/about-cancer/breast-cancer/stages-types-grades/tnm-staging
PC_FLATTEN_PTNM_TN_LABELS = True
VALID_PTNM_T_LABELS = ["TX", "T1", "T1a", "T1b", "T1c", "T2", "T2a", "T3", "T4", "T4b"]
PTNM_T_LABELS_HIERARCHY = {
    "TX": [],
    "T1": ["T1a", "T1b", "T1c"],
    "T2": ["T2a"],
    "T3": [],
    "T4": ["T4b"],
}
VALID_PTNM_N_LABELS = ["pNX", "pN0", "pN1", "pN1a", "pN1mi", "pN2", "pN2a", "pN3", "pN3a", "pN3b"]
PTNM_N_LABELS_HIERARCHY = {
    "pNX": [],
    "pN0": [],
    "pN1": ["pN1a", "pN1mi"],
    "pN2": ["pN2a"],
    "pN3": ["pN3a", "pN3b"],
}
VALID_PTNM_M_LABELS = ["M0", "cMo(i+)", "pM1"]


def get_description_of_cleaned_features():
    feature_description = {
        "image_level_features": [
            "FileName_FullStack",
            "merged_pid",
            "diseasestatus",
            "Height_FullStack",
            "Width_FullStack",
            "area",
            "sum_area_cells",
            "Count_Cells",
        ],
        "patient_level_features": [
            "PrimarySite",
            "Subtype",
            "clinical_type",
            "PTNM_T",
            "PTNM_N",
            "PTNM_M",
            "DFSmonth",
            "OSmonth",
            "images_per_patient",
            "images_per_patient_filtered",
            "cohort",
        ],
    }
    return feature_description


def get_predictive_clustering_valid_features():
    feature_description = get_description_of_cleaned_features()
    columns = (
        feature_description["image_level_features"] + feature_description["patient_level_features"]
    )
    columns = [
        s
        for s in columns
        if s
        not in [
            "FileName_FullStack",
            "merged_pid",
            "PrimarySite",
            "Height_FullStack",
            "Width_FullStack",
            "images_per_patient",
        ]
    ]
    return columns


def flatten_ptnm_tn_labels(my_class, ptnm_labels_hierarchy):
    if type(my_class) == float and math.isnan(my_class):
        return my_class
    big_classes_labels = list(ptnm_labels_hierarchy.keys())
    small_classes_labels = []
    for v in ptnm_labels_hierarchy.values():
        small_classes_labels.extend(v)

    if my_class in big_classes_labels:
        return my_class
    else:
        assert my_class in small_classes_labels
        parent_class = [k for k, v in ptnm_labels_hierarchy.items() for vv in v if vv == my_class]
        assert len(parent_class) == 1
        parent_class = parent_class[0]
        return parent_class


def clean_metadata(df_basel, df_zurich, verbose=False):
    if verbose:
        print("clearing metadata")
    df = pd.concat([df_basel, df_zurich])

    assert np.sum(df["FileName_FullStack"].isna()) == 0
    assert df["FileName_FullStack"].is_unique

    assert np.sum(df["PID"].isna()) == 0
    pids_basel = df["PID"].unique()
    # if not CONTINUOUS_INTEGRATION:
    #     assert len(pids_basel) == max(pids_basel)
    pids_zurich = df["PID"].unique()
    # if not CONTINUOUS_INTEGRATION:
    #     assert len(pids_zurich) == max(pids_zurich)
    df_basel = df_basel.rename(columns={"PID": "merged_pid"})
    df_zurich["merged_pid"] = df_zurich["PID"].apply(lambda x: x + max(pids_basel))
    df_zurich.drop("PID", axis=1, inplace=True)
    df = pd.concat([df_basel, df_zurich])
    feature_description = get_description_of_cleaned_features()
    assert set(df.columns.to_list()) == set(
        feature_description["image_level_features"] + feature_description["patient_level_features"]
    )
    pids = sorted(df["merged_pid"].unique())
    for my_pid in pids:
        dff = df.loc[df["merged_pid"] == my_pid, :]
        must_be_shared = feature_description["patient_level_features"]
        dfff = dff[must_be_shared]
        dfff.eq(dfff.iloc[0, :], axis=1)
        pass

    assert np.sum(df_basel["diseasestatus"].isna()) == 0
    assert np.sum(df_zurich["diseasestatus"].isna()) == len(df_zurich)
    df_zurich["diseasestatus"] = ["tumor"] * len(df_zurich)
    df = pd.concat([df_basel, df_zurich])
    assert all([e in DISEASE_STATUSES for e in df["diseasestatus"].value_counts().to_dict().keys()])

    assert df_basel["PrimarySite"].value_counts().to_dict() == {"breast": len(df_basel)}
    assert np.sum(df_zurich["PrimarySite"].isna()) == len(df_zurich)
    df_zurich["PrimarySite"] = ["breast"] * len(df_zurich)

    def warn_on_na(df, df_name, column):
        n = np.sum(df[column].isna())
        if n > 0:
            if verbose:
                print(f"warning: {df_name}[{column}] contains {n} NAs out of {len(df)} values")

    assert all([e in CANCER_SUBTYPES for e in df["Subtype"].value_counts().to_dict().keys()])
    warn_on_na(df_basel, "df_basel", "Subtype")
    warn_on_na(df_zurich, "df_zurich", "Subtype")

    assert all(
        [e in CANCER_CLINICAL_TYPES for e in df["clinical_type"].value_counts().to_dict().keys()]
    )
    warn_on_na(df_basel, "df_basel", "clinical_type")
    warn_on_na(df_zurich, "df_zurich", "clinical_type")

    assert np.sum(df["Height_FullStack"].isna()) == 0

    assert np.sum(df["Width_FullStack"].isna()) == 0

    assert np.sum(df["area"].isna()) == 0

    assert np.sum(df["sum_area_cells"].isna()) == 0

    assert np.sum(df["Count_Cells"].isna()) == 0

    # PTNM_T
    assert np.sum(df["PTNM_T"].isna()) == 0

    def ptnm_t_renamer(bad_label):
        label = re.sub(r"^t", "", bad_label)
        if label == "[]":
            label = "X"
        label = "T" + label
        return label

    df_basel["PTNM_T"] = df_basel["PTNM_T"].apply(ptnm_t_renamer)
    df_zurich["PTNM_T"] = df_zurich["PTNM_T"].apply(ptnm_t_renamer)
    if PC_FLATTEN_PTNM_TN_LABELS:
        if verbose:
            print("flattening PTNM_T labels")
        df_basel["PTNM_T"] = df_basel["PTNM_T"].apply(
            lambda x: flatten_ptnm_tn_labels(x, PTNM_T_LABELS_HIERARCHY)
        )
        df_zurich["PTNM_T"] = df_zurich["PTNM_T"].apply(
            lambda x: flatten_ptnm_tn_labels(x, PTNM_T_LABELS_HIERARCHY)
        )
    df = pd.concat([df_basel, df_zurich])
    assert all([e in VALID_PTNM_T_LABELS for e in df["PTNM_T"].value_counts().to_dict().keys()])
    if verbose:
        # TODO: check if this interpretation is correct
        print('warning: interpreting the PTNM_T label "[]" as "TX"')

    # PTNM_N
    assert np.sum(df["PTNM_N"].isna()) == 0

    def ptnm_n_renamer(bad_label):
        label = re.sub(r"^n", "", bad_label)
        if label in ["0sl", "0sn"]:
            label = "0"
        if label == "x" or label == "X" or label == "[]":
            label = "X"
        label = "pN" + label
        return label

    df_basel["PTNM_N"] = df_basel["PTNM_N"].apply(ptnm_n_renamer)
    df_zurich["PTNM_N"] = df_zurich["PTNM_N"].apply(ptnm_n_renamer)
    if PC_FLATTEN_PTNM_TN_LABELS:
        if verbose:
            print("flattening PTNM_N labels")
        df_basel["PTNM_N"] = df_basel["PTNM_N"].apply(
            lambda x: flatten_ptnm_tn_labels(x, PTNM_N_LABELS_HIERARCHY)
        )
        df_zurich["PTNM_N"] = df_zurich["PTNM_N"].apply(
            lambda x: flatten_ptnm_tn_labels(x, PTNM_N_LABELS_HIERARCHY)
        )
    df = pd.concat([df_basel, df_zurich])
    assert all([e in VALID_PTNM_N_LABELS for e in df["PTNM_N"].value_counts().to_dict().keys()])
    if verbose:
        # TODO: check if these interpretations are correct
        print('warning: interpreting the PTNM_N label "[]" as "pNX"')
        print('warning: interpreting the PTNM_N labels "0sl" and "0sn" as "pN0"')

    # PTNM_M
    assert np.sum(df["PTNM_M"].isna()) == 0

    def ptnm_m_renamer(bad_label):
        label = bad_label
        if label == "0" or label == 0:
            label = "M0"
        if label == "1" or label == 1 or label == "M1":
            label = "pM1"
        if label == "M0_IPLUS":
            label = "cMo(i+)"
        return label

    df_basel["PTNM_M"] = df_basel["PTNM_M"].apply(ptnm_m_renamer)
    df_zurich["PTNM_M"] = df_zurich["PTNM_M"].apply(ptnm_m_renamer)
    df = pd.concat([df_basel, df_zurich])
    assert all([e in VALID_PTNM_M_LABELS for e in df["PTNM_M"].value_counts().to_dict().keys()])
    if verbose:
        # TODO: check if this interpretations is correct
        print('warning: interpreting the "M0_IPLUS" label as "cMo(i+)"')

    assert np.sum(df_basel["DFSmonth"].isna()) == 0
    assert np.sum(df_zurich["DFSmonth"].isna()) == len(df_zurich)

    assert np.sum(df_basel["OSmonth"].isna()) == 0
    assert np.sum(df_zurich["OSmonth"].isna()) == len(df_zurich)
    if verbose:
        print("metadata cleaned")
    return df_basel, df_zurich


def get_metadata(basel_csv, zurich_csv, clean=True, clean_verbose=True):
    df_basel = pd.read_csv(basel_csv)
    df_zurich = pd.read_csv(zurich_csv)
    selected_columns = [
        "FileName_FullStack",
        "PID",
        "diseasestatus",
        "PrimarySite",
        "Subtype",
        "clinical_type",
        "Height_FullStack",
        "Width_FullStack",
        "area",
        "sum_area_cells",
        "Count_Cells",
        "PTNM_T",
        "PTNM_N",
        "PTNM_M",
        "DFSmonth",
        "OSmonth",
    ]
    df_basel = df_basel[selected_columns]
    df_zurich = df_zurich[selected_columns]

    df_basel["images_per_patient"] = (
        df_basel["FileName_FullStack"].groupby(df_basel["PID"]).transform("count")
    )
    df_zurich["images_per_patient"] = (
        df_zurich["FileName_FullStack"].groupby(df_zurich["PID"]).transform("count")
    )

    valid_omes = df_basel.FileName_FullStack.to_list() + df_zurich.FileName_FullStack.to_list()

    dropped_basel = len(df_basel[~df_basel["FileName_FullStack"].isin(valid_omes)])
    df_basel = df_basel[df_basel["FileName_FullStack"].isin(valid_omes)]
    print(f"discarding {dropped_basel} omes from the Basel cohort, remaining: {len(df_basel)}")
    # assert dropped_basel == 0

    dropped_zurich = len(df_zurich[~df_zurich["FileName_FullStack"].isin(valid_omes)])
    df_zurich = df_zurich[df_zurich["FileName_FullStack"].isin(valid_omes)]
    print(f"discarding {dropped_basel} omes from the Zurich cohort, remaning: {len(df_zurich)}")
    # assert dropped_zurich == 0

    df_basel["images_per_patient_filtered"] = (
        df_basel["FileName_FullStack"].groupby(df_basel["PID"]).transform("count")
    )
    df_zurich["images_per_patient_filtered"] = (
        df_zurich["FileName_FullStack"].groupby(df_zurich["PID"]).transform("count")
    )
    df_basel["cohort"] = "basel"
    df_zurich["cohort"] = "zurich"

    # better safe than sorry
    assert len(df_basel) + len(df_zurich) == len(valid_omes)

    if clean:
        df_basel, df_zurich = clean_metadata(df_basel, df_zurich, verbose=clean_verbose)

    df = pd.concat([df_basel, df_zurich])
    return df


def create_muon_spatial_object(f_ome, f_masks, outfile):
    ome = tifffile.TiffFile(f_ome)
    metadata = ElementTree.fromstring(ome.ome_metadata)[0]
    for chld in metadata:
        if chld.tag.endswith("Pixels"):
            metadata = chld
            break
    channel_names = []
    for channel in metadata:
        if channel.tag.endswith("Channel"):
            channel_names.append(channel.attrib["Fluor"])

    masks = np.asarray(tifffile.imread(f_masks))
    if os.path.isfile(outfile):
        os.unlink(outfile)
    smudata = spatialmuon.SpatialMuData(outfile)
    smudata["IMC"] = modality = spatialmuon.SpatialModality(coordinate_unit="μm")
    modality["FOV"] = fov = spatialmuon.Raster(
        X=np.moveaxis(ome.asarray(), 0, -1), channel_names=channel_names
    )
    fov.feature_masks["cells"] = spatialmuon.RasterMask(mask=masks)


if len(sys.argv) > 1:
    outfdir = sys.argv[1]
else:
    outfdir = "imc"

with tempfile.TemporaryDirectory() as tmpdir:
    imgfile = os.path.join(tmpdir, "images.zip")
    download(
        "https://zenodo.org/record/3518284/files/OMEandSingleCellMasks.zip?download=1",
        imgfile,
        desc="images",
    )
    print("extracting images...", file=sys.stderr)
    unzip(imgfile, tmpdir)
    unzip(os.path.join(tmpdir, "OMEnMasks", "ome.zip"), tmpdir)
    unzip(os.path.join(tmpdir, "OMEnMasks", "Basel_Zuri_masks.zip"), tmpdir)

    metadatafile = os.path.join(tmpdir, "metadata.zip")
    download(
        "https://zenodo.org/record/3518284/files/SingleCell_and_Metadata.zip?download=1",
        metadatafile,
        desc="metadata",
    )
    unzip(
        metadatafile,
        tmpdir,
        [
            "Data_publication/BaselTMA/Basel_PatientMetadata.csv",
            "Data_publication/ZurichTMA/Zuri_PatientMetadata.csv",
        ],
    )

    df = get_metadata(
        clean_verbose=True,
        basel_csv=os.path.join(tmpdir, "Data_publication", "BaselTMA", "Basel_PatientMetadata.csv"),
        zurich_csv=os.path.join(
            tmpdir, "Data_publication", "ZurichTMA", "Zuri_PatientMetadata.csv"
        ),
    )

    for index, row in tqdm(df.iterrows(), total=len(df), desc="creating .hs5mu objects"):
        ome_filename = row[0]
        f_ome = os.path.join(tmpdir, "ome", ome_filename)

        masks_filename = ome_filename.replace("_full.tiff", "_full_maks.tiff")
        f_masks = os.path.join(tmpdir, "Basel_Zuri_masks", masks_filename)

        output_folder = "generated_muon"
        os.makedirs(output_folder, exist_ok=True)
        outfile = ome_filename.replace(".tiff", ".h5smu")
        outfile = os.path.join(output_folder, outfile)
        create_muon_spatial_object(f_ome, f_masks, outfile)
