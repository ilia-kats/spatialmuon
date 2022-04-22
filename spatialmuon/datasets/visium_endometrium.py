##
import tempfile

import spatialmuon
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import gc

root = "/data/spatialmuon/datasets/visium_endometrium/"

samples = ["152806", "152807", "152810", "152811"]


def get_smu_file(sample):
    f = os.path.join(root, f"smu/{sample}.h5smu")
    s = spatialmuon.SpatialMuData(f)
    return s


def create_smu_files(sample, hires_images=False):
    converter = spatialmuon.Converter()
    # somehow I can clean the memory and I need to run the script different times for different samples, otherwise I
    # run out of RAM
    # samples = samples[:2]
    # samples = samples[2:]
    sample_dir = os.path.join(root, "raw/visium", sample)
    mod = converter.read_visium10x(sample_dir)

    Image.MAX_IMAGE_PIXELS = 5000000000

    def get_hires_img(sample):
        f = "/data/spatialmuon/datasets/visium_endometrium/raw/hires_images"
        if sample == "152810":
            res = "20x"
        else:
            res = "40x"
        path = os.path.join(f, f"{sample}_{res}_highest_res_image.jpg")
        im = Image.open(path)
        return np.array(im)

    s = spatialmuon.SpatialMuData(os.path.join(root, f"smu/{sample}.h5smu"), backingmode="w")
    s["visium"] = mod
    if hires_images:
        # start = time.time()
        hires = get_hires_img(sample)
        # print(f'reading the image: {time.time() - start}')

        # start = time.time()
        assert hires.dtype == np.dtype("uint8")
        assert np.max(hires) > 1

        if sample == "152806":
            source_points = np.array([[19465, 50704], [36663, 12568]])
            target_points = np.array([[597, 1718], [1164, 462]])
        elif sample == "152807":
            source_points = np.array([[4675, 19252], [26610, 36954]])
            target_points = np.array([[172, 695], [913, 1294]])
        elif sample == "152810":
            source_points = np.array([[2945, 10284], [17174, 23167]])
            target_points = np.array([[152, 730], [1090, 1579]])
        elif sample == "152811":
            source_points = np.array([[29704, 41867], [4809, 21925]])
            target_points = np.array([[1033, 1458], [190, 783]])
        else:
            raise ValueError()

        anchor = spatialmuon.Anchor.map_untransformed_to_untransformed_fov(
            s["visium"]["image"], source_points=source_points, target_points=target_points
        )
        raster = spatialmuon.Raster(
            X=hires,
            coordinate_unit=s["visium"]["image"].coordinate_unit,
            anchor=anchor,
        )
        s["visium"]["hires_image"] = raster
        # print(f'writing the image: {time.time() - start}')

        # start = time.time()
        # getting out of RAM :(
        s._backing.close()
        del s
        gc.collect()
        # print(f'collecting garbage: {time.time() - start}')


def show(sample):
    f = os.path.join(root, f"smu/{sample}.h5smu")
    s = spatialmuon.SpatialMuData(f)

    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 10))
    # bb = None
    bb = spatialmuon.BoundingBox(x0=1800, x1=2200, y0=3800, y1=5000)
    s["visium"]["image"].plot(ax=ax0, bounding_box=bb)
    s["visium"]["expression"].plot(
        0, fill_color=None, outline_color="channel", ax=ax0, bounding_box=bb
    )
    s["visium"]["expression"].set_lims_to_bounding_box(bb, ax=ax0)

    start = time.time()
    s["visium"]["hires_image"].plot(ax=ax1, bounding_box=bb)
    print(f"plotting the hires image: {time.time() - start}")
    start = time.time()
    s["visium"]["expression"].plot(
        0, fill_color=None, outline_color="channel", ax=ax1, bounding_box=bb
    )
    print(f"plotting all the spots: {time.time() - start}")
    s["visium"]["expression"].set_lims_to_bounding_box(bb, ax=ax1)
    plt.show()


def downscale(sample):
    f = os.path.join(root, f"smu/{sample}.h5smu")
    s = spatialmuon.SpatialMuData(f)
    start = time.time()
    f = s["visium"]["hires_image"].clone()
    print(f"cloning the large image: {time.time() - start}")
    bb = spatialmuon.BoundingBox(x0=2000, x1=2200, y0=4000, y1=4200)
    # bb = None

    axes = plt.subplots(1, 2, figsize=(15, 5))[1].flatten()
    f.plot(bounding_box=bb, ax=axes[0])
    target_scale_factor = 2
    k = target_scale_factor / s["visium"]["hires_image"].anchor.scale_factor
    original_w = s["visium"]["hires_image"].X.shape[1]
    target_w = original_w * k
    print(f"original_w = {original_w}, target_w = {target_w}")
    f.scale_raster(target_w=target_w)
    f.plot(bounding_box=bb, ax=axes[1])
    plt.show()

    if "medium_res" in s["visium"]:
        del s["visium"]["medium_res"]
    s["visium"]["medium_res"] = f
    print(s)


def make_tiles(sample):
    f = os.path.join(root, f"smu/{sample}.h5smu")
    s = spatialmuon.SpatialMuData(f)
    # tiles = s['visium']['w5000'].extract_tiles(masks=s['visium']['expression'].masks, tile_dim_in_pixels=50)
    start = time.time()
    tiles = s["visium"]["medium_res"].extract_tiles(
        masks=s["visium"]["expression"].masks, tile_dim_in_units=55
    )
    print(f"extracting large tiles: {time.time() - start}")
    len(tiles.tiles)

    # finding indices of tiles intersecting the interior of the bounding box
    bb = spatialmuon.BoundingBox(x0=1000, x1=1500, y0=4000, y1=4500)
    m = s["visium"]["expression"].masks
    subset = m.crop(bounding_box=bb)
    indices = subset.obs.index

    # plotting the tiles aligned to the original image (which is plotted in the background with transparency)
    _, ax = plt.subplots(1, figsize=(10, 10))
    s["visium"]["hires_image"].plot(ax=ax, alpha=0.3, bounding_box=bb)
    for i in indices:
        # there is a problem with this tile
        # if i == 606:
        #     print("ooo")
        #     print("ooo")
        t = tiles.tile_to_raster(i)
        t.plot(ax=ax, bounding_box=bb, show_scalebar=False)
    s["visium"]["expression"].set_lims_to_bounding_box(bb=bb, ax=ax)
    plt.show()

    axes = plt.subplots(5, 5, figsize=(15, 15))[1].flatten()
    for i, ax in enumerate(axes):
        t = tiles.tiles[i]
        ax.imshow(t)
    plt.show()


def add_cell2location_data():
    cell2location_data_folder = os.path.join(
        root,
        "raw/visium/20201207_LocationModelLinearDependentWMultiExperiment_19clusters_20952locations_19980genes",
    )
    os.listdir(cell2location_data_folder)
    f = os.path.join(cell2location_data_folder, "sp.h5ad")
    import anndata as ad
    import scanpy as sc

    a = ad.read_h5ad(f)
    ##
    adatas = {}
    for sample in samples:
        ii = a.obs["sample"] == sample
        aa = a[ii]
        adatas[sample] = aa
    ##
    cell_types = [
        "Endothelial ACKR1",
        "Endothelial SEMA3G",
        "Epithelial Ciliated",
        "Epithelial Ciliated LRG5",
        "Epithelial Glandular",
        "Epithelial Glandular_secretory",
        "Epithelial Lumenal 1",
        "Epithelial Lumenal 2",
        "Epithelial Pre-ciliated",
        "Epithelial SOX9",
        "Epithelial SOX9_LGR5",
        "Fibroblast C7",
        "Fibroblast dS",
        "Fibroblast eS",
        "Lymphoid",
        "Myeloid",
        "PV MYH11",
        "PV STEAP4",
        "uSMC",
    ]
    for sample in tqdm(samples):
        for feature in ["mean_spot_factors", "mean_nUMI_factors"]:
            s = get_smu_file(sample)
            aa = adatas[sample]
            spots = s["visium"]["expression"]
            # print(f'len(aa) = {len(aa)}, len(spots.obs) = {len(spots.obs)}')
            assert len(aa) == len(spots.obs)
            original_labels = aa.obs["spot_id"].to_numpy()
            smu_labels = spots.masks.masks_labels
            s0 = set(original_labels)
            s1 = set(smu_labels)
            different = s0.symmetric_difference(s1)
            assert len(different) == 0

            columns = [f"{feature}{c}" for c in cell_types]

            df = aa.obs[columns].copy()
            df["original_label"] = original_labels
            df.set_index(keys="original_label", inplace=True)

            data_for_smu = df.loc[smu_labels]
            import pandas as pd

            obs = pd.DataFrame(index=original_labels)
            var = pd.DataFrame({"channel_name": columns})
            masks = s["visium"]["expression"].masks.clone()
            smu_factors = spatialmuon.Regions(
                X=data_for_smu.to_numpy(),
                var=var,
                masks=masks,
                anchor=s["visium"]["expression"].anchor,
            )
            if feature in s["visium"]:
                del s["visium"][feature]
            s["visium"][feature] = smu_factors

            _, axes = plt.subplots(1, 2)
            ch = f"{feature}Endothelial ACKR1"
            s["visium"]["image"].plot(ax=axes[0])
            smu_factors.plot(ch, ax=axes[0])
            sc.pl.spatial(aa, color=ch, spot_size=100, ax=axes[1])
            plt.show()
            s.backing.close()
        ##


if __name__ == "__main__":
    ##
    for sample in tqdm(samples, desc="creating .h5smu"):
        create_smu_files(sample, hires_images=True)
        show(sample)
    ##
    for sample in tqdm(samples, desc="downscaling"):
        downscale(sample)
    ##
    if False:
        for sample in samples:
            s = get_smu_file(sample)
            print(s["visium"]["medium_res"].anchor.scale_factor)
            make_tiles(sample)
    ##
    add_cell2location_data()
    for sample in samples:
        s = get_smu_file(sample)
        print(s)
        s.backing.close()
