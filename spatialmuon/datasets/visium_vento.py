##
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
            hires = hires / 255

            source_points = None
            target_points = None
            if sample == '152807':
                source_points = np.array([[4675, 19252], [26610, 36954]])
                target_points = np.array([[172, 695], [913, 1294]])

            assert source_points is not None and target_points is not None
            anchor = spatialmuon.Anchor.map_untransformed_to_untransformed_fov(
                s["visium"]["image"],
                source_points=source_points,
                target_points=target_points
            )
            raster = spatialmuon.Raster(X=hires, coordinate_unit=ss[sample]['visium']['image'].coordinate_unit,
                                        anchor=anchor)
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
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 20))
    bb = spatialmuon.BoundingBox(x0=2000, x1=2200, y0=4000, y1=4200)
    s["visium"]["image"].plot(ax=ax0, bounding_box=bb)
    s["visium"]["expression"].plot(0, ax=ax0, bounding_box=bb)
    s["visium"]["expression"].set_lims_to_bounding_box(bb, ax=ax0)
    # ##
    # s["visium"]["image"].plot(ax=ax1)
    # s["visium"]["expression"].plot(0, ax=ax1)
    # ax1.set(xlim=(2000, 2200), ylim=(4000, 4200))
    ##
    bb = spatialmuon.BoundingBox(x0=2000, x1=2200, y0=4000, y1=4200)
    start = time.time()
    s["visium"]["hires_image"].plot(ax=ax1, bounding_box=bb)
    print(f"plotting the hires image: {time.time() - start}")
    start = time.time()
    s["visium"]["expression"].plot(0, ax=ax1, bounding_box=bb)
    print(f"plotting all the spots: {time.time() - start}")
    s["visium"]["expression"].set_lims_to_bounding_box(bb, ax=ax1)
    plt.show()
    # ##
    # fov = s['visium']['hires_image']
    # # fov._anchor = anchor
    # # fov.plot()
    # crop = fov.X[:20000, :5000, :]
    # if 'crop' in s['visium']:
    #     del s['visium']['crop']
    # s['visium']['crop'] = spatialmuon.Raster(X=crop, anchor=anchor, coordinate_unit='um')
    # s_crop = s['visium']['crop']
    # _, ax = plt.subplots(1)
    # s['visium']['image'].plot(ax=ax)
    # s_crop.plot(ax=ax)
    # plt.show()
    ##
    pass


if __name__ == "__main__":
    # create_smu_files(hires_images=True)
    show()
