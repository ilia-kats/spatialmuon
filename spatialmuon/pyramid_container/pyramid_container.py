# THIS FILE IS ONLY A PROTOTYPE

import os
import PIL
import PIL.Image
from typing import Literal
from os import PathLike
from skimage.transform import pyramid_gaussian

from typing import Optional, Union, List

import numpy as np
import h5py
import xarray_multiscale
from tqdm import tqdm


class PyramidContainer:
    def __init__(
        self,
        file: str,
        file_mode: Literal["r", "r+", "w"] = "r+",
        array: Optional[np.ndarray] = None,
    ):
        if file_mode == "r":
            assert array is None
            if not os.path.isfile(file):
                raise FileNotFoundError(
                    f"file {file} not found; please ensure that a file/symilik is available at that location"
                )
        if file_mode == "r+" and os.path.isfile(file):
            assert array is None
        if file_mode == "r+" and not os.path.isfile(file):
            assert array is not None
        if file_mode == "w":
            assert array is not None

        self.file = file
        self.file_mode = file_mode

        if array is None:
            self.load()
        else:
            self.save(array)

    def load(self):
        pass

    def save(self, array):
        # from xarray_multiscale import multiscale
        # from xarray_multiscale.reducers import windowed_mean
        #
        # assert len(array.shape) == 3
        # channels = []
        # for c in tqdm(range(array.shape[-1]), desc='creating pyramids'):
        #     data = array[:, :, c]
        #     p = multiscale(array=data, reduction=windowed_mean, scale_factors=(2, 2))
        #     channels.append(p)
        pass


if __name__ == "__main__":
    large_img = "/data/spatialmuon/datasets/visium_endometrium/raw/hires_images/152806_40x_highest_res_image.jpg"
    PIL.Image.MAX_IMAGE_PIXELS = 5000000000
    print('loading a large image... ', end='')
    im = PIL.Image.open(large_img)
    array = np.array(im)
    print('done')
    ##
    import spatialmuon as smu
    with h5py.File('/data/l989o/temp/big_fov.h5smu', 'w') as f5:
        s = smu.Raster(X=array, backing=f5)
        pass
    ##
    # pyramid_container = PyramidContainer(
    #     file="/data/spatialmuon/datasets/visium_endometrium/smu/152806_pyramid",
    #     file_mode="w",
    #     array=array,
    # )
