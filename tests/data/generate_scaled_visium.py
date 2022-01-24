import spatialmuon as smu
from spatialmuon._core.anchor import Anchor
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import shutil

this_dir = Path(__file__).parent
infile = this_dir / "../data/small_visium.h5smu"
outfile = this_dir / "../data/scaled_visium.h5smu"
shutil.copy(infile, outfile)

s = smu.SpatialMuData(outfile)

image = s['visium']['image'].X[...]
im = Image.fromarray(image)

im2x = im.resize((im.width * 2, im.height * 2), Image.ANTIALIAS)
border = (30, 30, 30, 30) # left, top, right, bottom
im_crop = ImageOps.crop(im, border)
im2x_crop = ImageOps.crop(im2x, border)

origin = s['visium']['image'].anchor.origin[...]
scale_factor = s['visium']['image'].anchor.scale_factor
anchor2x = Anchor(origin=origin, vector=np.array([2., 0.]) * scale_factor)
anchor_crop = Anchor(origin=origin + np.array([30., 30.]) / scale_factor, vector=np.array([1., 0.]) * scale_factor)
anchor2x_crop = Anchor(origin=origin + np.array([15., 15.]) / scale_factor, vector=np.array([2., 0.]) * scale_factor)

mod = s['visium']
mod['image2x'] = smu.Raster(X=np.array(im2x), anchor=anchor2x)
mod['image_crop'] = smu.Raster(X=np.array(im_crop), anchor=anchor_crop)
mod['image2x_crop'] = smu.Raster(X=np.array(im2x_crop), anchor=anchor2x_crop)
#
# s_out['visium'] = new_mod
# # plt.figure()
# # plt.imshow(im)
# # plt.imshow(im2x)
# # plt.imshow(im_crop)
# # plt.show()
print(s)
print(f'created .h5smu file at {outfile}')