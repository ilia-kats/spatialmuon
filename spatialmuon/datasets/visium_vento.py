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


def create_smu_files(hires_images=False):

    converter = spatialmuon.Converter()
    samples = ["152806", "152810", "152811", "152807"]
    # somehow I can clean the memory and I need to run the script different times for different samples, otherwise I
    # run out of RAM
    # samples = samples[:2]
    # samples = samples[2:]
    samples = samples[3:]
    mods = {}
    for sample in samples:
        sample_dir = os.path.join(root, "raw/visium", sample)
        mod = converter.read_visium10x(sample_dir)
        mods[sample] = mod

    ##
    Image.MAX_IMAGE_PIXELS = 5000000000

    def get_hires_img(sample):
        f = "/data/spatialmuon/datasets/visium_endometrium/raw/hires_images"
        path = os.path.join(f, f"{sample}_40x_highest_res_image.jpg")
        im = Image.open(path)
        return np.array(im)

    ##
    ss = {}
    for sample in samples:
        s = spatialmuon.SpatialMuData(os.path.join(root, f"smu/{sample}.h5smu"), backingmode="w")
        s["visium"] = mods[sample]
        ss[sample] = s
    ##
    if hires_images:
        for sample in tqdm(samples, desc="processing large images"):
            # start = time.time()
            hires = get_hires_img(sample)
            # print(f'reading the image: {time.time() - start}')

            # start = time.time()
            assert hires.dtype == np.dtype("uint8")
            assert np.max(hires) > 1

            source_points = None
            target_points = None
            if sample == "152807":
                source_points = np.array([[4675, 19252], [26610, 36954]])
                target_points = np.array([[172, 695], [913, 1294]])

            assert source_points is not None and target_points is not None
            anchor = spatialmuon.Anchor.map_untransformed_to_untransformed_fov(
                s["visium"]["image"], source_points=source_points, target_points=target_points
            )
            raster = spatialmuon.Raster(
                X=hires,
                coordinate_unit=ss[sample]["visium"]["image"].coordinate_unit,
                anchor=anchor,
            )
            ss[sample]["visium"]["hires_image"] = raster
            # print(f'writing the image: {time.time() - start}')

            # start = time.time()
            # getting out of RAM :(
            ss[sample]._backing.close()
            del ss[sample]
            gc.collect()
            # print(f'collecting garbage: {time.time() - start}')


def show():
    f = os.path.join(root, "smu/152807.h5smu")
    s = spatialmuon.SpatialMuData(f)
    ##
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 10))
    bb = spatialmuon.BoundingBox(x0=2000, x1=2200, y0=4000, y1=4200)
    s["visium"]["image"].plot(ax=ax0, bounding_box=bb)
    s["visium"]["expression"].plot(
        0, fill_color=None, outline_color="channel", ax=ax0, bounding_box=bb
    )
    s["visium"]["expression"].set_lims_to_bounding_box(bb, ax=ax0)
    ##
    bb = spatialmuon.BoundingBox(x0=2000, x1=2200, y0=4000, y1=4200)
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
    ##
    pass


def downscale():
    f = os.path.join(root, "smu/152807.h5smu")
    s = spatialmuon.SpatialMuData(f)
    start = time.time()
    f = s["visium"]["hires_image"].clone()
    print(f"cloning the large image: {time.time() - start}")
    bb = spatialmuon.BoundingBox(x0=2000, x1=2200, y0=4000, y1=4200)

    axes = plt.subplots(1, 2, figsize=(15, 5))[1].flatten()
    f.plot(bounding_box=bb, ax=axes[0])
    f.scale_raster(target_w=5000)
    f.plot(bounding_box=bb, ax=axes[1])
    plt.show()

    # with tempfile.TemporaryDirectory() as td:
    #     f1 = os.path.join(td, 'a.h5smu')
    #     s1 = spatialmuon.SpatialMuData(f1)
    #     m = spatialmuon.SpatialModality()
    #     s1['a'] = m
    #     m['a'] = f
    #     print('ooo')
    #     print('ooo')
    s["visium"]["w5000"] = f


def make_tiles():
    f = os.path.join(root, "smu/152807.h5smu")
    s = spatialmuon.SpatialMuData(f)
    # tiles = s['visium']['w5000'].extract_tiles(masks=s['visium']['expression'].masks, tile_dim_in_pixels=50)
    start = time.time()
    tiles = s["visium"]["hires_image"].extract_tiles(
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
        if i == 606:
            print("ooo")
            print("ooo")
        t = tiles.tile_to_raster(i)
        t.plot(ax=ax, bounding_box=bb)
    s["visium"]["expression"].set_lims_to_bounding_box(bb=bb, ax=ax)
    plt.show()
    ##
    axes = plt.subplots(5, 5, figsize=(15, 15))[1].flatten()
    for i, ax in enumerate(axes):
        t = tiles.tiles[i]
        ax.imshow(t)
    plt.show()
    #l


if __name__ == "__main__":
    # create_smu_files(hires_images=True)
    # show()
    # downscale()
    make_tiles()
