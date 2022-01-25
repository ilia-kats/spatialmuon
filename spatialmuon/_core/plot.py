# legacy code, to be gradually included into the new code codebase

from __future__ import annotations

from spatialmuon.utils import angle_between
from typing import Union, Optional, Callable, List
import spatialmuon.datatypes
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.image
import matplotlib.patches
import matplotlib.collections
import matplotlib.transforms
import math
import warnings

from scipy import ndimage

import numpy as np


def plot_channel_raster(
    fov: FieldOfView,
    ax: matplotlib.axes.Axes,
    channel: int,
    preprocessing: Callable,
    alpha: float,
    color: Optional[Union[tuple[float], str]],
    cmap: Optional[matplotlib.colors.Colormap],
    **kwargs,
) -> matplotlib.image.AxesImage:
    x = fov.X[...][:, :, channel]
    if preprocessing is not None:
        x = preprocessing(x)
    if color is not None:
        raise NotImplementedError()
    if fov.ndim != 2:
        raise NotImplementedError("Can only do 2D for now")
    v1 = np.array((1, 0))
    v2 = fov.vector
    deg = angle_between(v1, v2)

    w, h = x.shape
    cx, cy = fov.origin

    if not (int(cx) == 0 and int(cy) == 0):
        x = np.pad(x, pad_width=((cx, 0), (cy, 0)))
    if deg != 0:
        pad_x = int(w - cx)
        pad_y = int(h - cy)
        print(pad_x, pad_y, cx, cy)
        # TODO(ttreis): Implement padding if centerpoint is right/bottom
        x_pad = np.pad(x, pad_width=((pad_x, 0), (pad_y, 0)))
        x_pad_rot = ndimage.rotate(x_pad, deg, reshape=False)
        x_pad_rot_cut = x_pad_rot[pad_x:, pad_y:]
        x = np.pad(x_pad_rot_cut, pad_width=((int(cx), 0), (int(cy), 0)))

    im = ax.imshow(x, alpha=alpha, cmap=cmap, **kwargs)

    return im

