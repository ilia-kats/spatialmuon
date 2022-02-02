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


def plot_channel_array(
    fov: FieldOfView,
    ax: matplotlib.axes.Axes,
    channel: int,
    preprocessing: Callable,
    alpha: float,
    color: Optional[Union[tuple[float], str]],
    cmap: Optional[matplotlib.colors.Colormap],
    random_colors: bool,
    **kwargs,
) -> Optional[matplotlib.cm.ScalarMappable]:
    x = fov.X[...][:, channel]
    x = x.todense()
    if preprocessing is not None:
        x = preprocessing(x)
    if fov._spot_shape != spatialmuon.datatypes.array.SpotShape.circle:
        raise NotImplementedError("preliminary implementation for Visium-like data only")
    points = fov.obs["geometry"].tolist()
    coords = np.array([[p.x, p.y] for p in points])
    radius = fov._spot_size
    patches = []
    for xy in coords:
        patch = matplotlib.patches.Circle(xy, radius)
        patches.append(patch)
    collection = matplotlib.collections.PatchCollection(patches)
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # hack to check if this is a newly created empty plot
    if xlim == (0.0, 1.0) and ylim == (0.0, 1.0):
        new_xlim = (x_min, x_max)
        new_ylim = (y_min, y_max)
    else:
        new_xlim = (min(xlim[0], x_min), max(xlim[1], x_max))
        new_ylim = (min(ylim[0], y_min), max(ylim[1], y_max))
    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)
    ax.set_aspect("equal")
    if random_colors:
        collection.set_color(np.random.rand(len(x), 3))
        scalar_mappable = None
    elif color is not None:
        collection.set_color(color)
        scalar_mappable = None
    else:
        colors = []
        a = x.min()
        b = x.max()
        norm = matplotlib.colors.Normalize(vmin=a, vmax=b)
        scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        for row, _ in enumerate(patches):
            value = x[row].item()
            # color = cmap((value - a) / (b - a))
            colors.append(value)
        collection.set_array(np.array(colors))
    collection.set_alpha(alpha)
    ax.add_collection(collection, autolim=False)
    return scalar_mappable


def image_to_rgb(x: np.ndarray):
    assert len(x.shape) == 3
    assert x.shape[-1] == 3
    old_shape = x.shape
    new_shape = (x.shape[0] * x.shape[1], 3)
    x.shape = new_shape
    a = np.min(x, axis=0)
    b = np.max(x, axis=0)
    x = (x - a) / (b - a)
    x.shape = old_shape
    return x
