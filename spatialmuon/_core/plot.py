from __future__ import annotations

from spatialmuon.utils import angle_between
from typing import Union, Optional, Callable
import spatialmuon.datatypes
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.image
import matplotlib.patches
import matplotlib.collections
import matplotlib.transforms
from matplotlib.transforms import Affine2D
import math
import warnings

from scipy import ndimage
# import matplotlib.font_manager as fm)
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np


def plot_channel(
    sm: SpatialModality,
    fov_name: str,
    ax: matplotlib.axes.Axes,
    *,
    channel: Optional[Union[str, int]] = None,
    preprocessing: Optional[Callable] = None,
    alpha: float = 1.0,
    color: Optional[Union[tuple[float], str]] = None,
    cmap: Optional[matplotlib.colors.Colormap] = matplotlib.cm.viridis,
    random_colors: bool = False,
    scalebar=True,
    **kwargs,
) -> Optional[matplotlib.cm.ScalarMappable]:
    if fov_name not in sm:
        raise ValueError(
            f"{fov_name} is not a FieldOfView of the considered SpatialModality"
        )
    else:
        fov = sm[fov_name]

    if random_colors:
        assert type(fov) != spatialmuon.datatypes.raster.Raster
        color = None
        cmap = None
    else:
        if color is not None:
            cmap = None
        else:
            if cmap is None:
                raise ValueError()

    if channel is not None:
        if type(channel) == str:
            # inefficient for many repeated calls
            if type(fov) == spatialmuon.datatypes.array.Array:
                if channel not in fov.var.index:
                    raise ValueError("channel not found")
                else:
                    i = fov.var.index.get_loc(channel)
            elif type(fov) == spatialmuon.datatypes.raster.Raster:
                if channel not in fov.channel_names:
                    raise ValueError("channel not found")
                else:
                    i = fov.channel_names.index(channel)
            else:
                raise ValueError()
        elif type(channel) == int:
            i = channel
        else:
            raise ValueError()
    else:
        raise ValueError("channel is None")

    if type(fov) == spatialmuon.datatypes.raster.Raster:
        scalar_mappable = plot_channel_raster(
            fov,
            ax,
            channel=i,
            preprocessing=preprocessing,
            alpha=alpha,
            color=color,
            cmap=cmap,
            **kwargs,
        )
    elif type(fov) == spatialmuon.datatypes.array.Array:
        scalar_mappable = plot_channel_array(
            fov,
            ax,
            channel=i,
            preprocessing=preprocessing,
            alpha=alpha,
            color=color,
            cmap=cmap,
            random_colors=random_colors,
            **kwargs,
        )
    else:
        raise NotImplementedError()

    if scalebar:
        unit = "um" if "μ" in sm.coordinate_unit else sm.coordinate_unit
        scalebar = ScaleBar(fov.scale, unit, box_alpha=0.8)
        ax.add_artist(scalebar)

    return scalar_mappable


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
        pad_x = int(w-cx)
        pad_y = int(h-cy)
        print(pad_x, pad_y, cx, cy)
        # TODO(ttreis): Implement padding if centerpoint is right/bottom
        x_pad = np.pad(x, pad_width=((pad_x, 0), (pad_y, 0)))
        x_pad_rot = ndimage.rotate(x_pad, deg, reshape=False)
        x_pad_rot_cut = x_pad_rot[pad_x:, pad_y:]
        x = np.pad(
            x_pad_rot_cut, 
            pad_width=((int(cx), 0), (int(cy), 0))
        )
        
    im = ax.imshow(
        x, 
        alpha=alpha,
        cmap=cmap,
        **kwargs
    )
    
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
        raise NotImplementedError(
            "preliminary implementation for Visium-like data only"
        )
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


def plot_image(
    sm: SpatialModality,
    fov_name: str,
    ax: matplotlib.axes.Axes,
    rgb_channels: Optional[Union[list[str], list[int]]] = None,
    preprocessing: Optional[Callable] = None,
    scalebar=True,
    **kwargs,
):
    if fov_name not in sm:
        raise ValueError(
            f"{fov_name} is not a FieldOfView of the considered SpatialModality"
        )
    else:
        fov = sm[fov_name]
    if type(fov) == spatialmuon.datatypes.raster.Raster:
        plot_image_raster(fov, ax, rgb_channels, preprocessing, **kwargs)
    else:
        raise NotImplementedError()

    if scalebar:
        unit = "um" if "μ" in sm.coordinate_unit else sm.coordinate_unit
        scalebar = ScaleBar(fov.scale, unit, box_alpha=0.8)
        ax.add_artist(scalebar)


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


def plot_image_raster(
    fov: FieldOfView,
    ax: matplotlib.axes.Axes,
    rgb_channels: Optional[Union[list[str], list[int]]],
    preprocessing: Optional[Callable],
    **kwargs,
) -> Optional[matplotlib.image.AxesImage]:
    n = fov.X.shape[-1]
    if rgb_channels is None:
        if n != 3:
            raise ValueError("interpreting only tensors with 3 channels as rgb")
        else:
            i = np.array([0, 1, 2])
    else:
        if type(rgb_channels) == list and type(rgb_channels[0]) == str:
            i = np.where(np.array(fov.channel_names) == rgb_channels)[0]
        elif type(rgb_channels) == int:
            i = np.array(rgb_channels)
        else:
            raise ValueError()
    x = fov.X[...][:, :, i]
    if preprocessing is not None:
        x = preprocessing(x)
    im = ax.imshow(x, **kwargs)
    return im

def plot_preview_grid(
      data_to_plot: dict = None,
      grid_size: Union[int, list[int]] = 1,
      preprocessing: Optional[Callable] = None,
      overlap: bool = False,
      cmap: Union[matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]] = matplotlib.cm.get_cmap("viridis")
    ):
    
    plt.style.use("dark_background")
    default_cmaps = [
        matplotlib.cm.get_cmap("viridis"),
        matplotlib.cm.get_cmap("plasma"),
        matplotlib.cm.get_cmap("inferno"),
        matplotlib.cm.get_cmap("magma"),
        matplotlib.cm.get_cmap("cividis"),
        matplotlib.cm.get_cmap("Purples"),
        matplotlib.cm.get_cmap("Blues"),
        matplotlib.cm.get_cmap("Greens"),
        matplotlib.cm.get_cmap("Oranges"),
        matplotlib.cm.get_cmap("Reds"),
        matplotlib.cm.get_cmap("spring"),
        matplotlib.cm.get_cmap("summer"),
        matplotlib.cm.get_cmap("autumn"),
        matplotlib.cm.get_cmap("winter"),
        matplotlib.cm.get_cmap("cool"),
    ]
    upper_limit_tiles = 50
    default_grid = [5, 5]

    if isinstance(grid_size, list) and len(grid_size) == 2 and all(isinstance(x, int) for x in grid_size):
        n_tiles = grid_size[0] * grid_size[1]
    elif grid_size == 1 and len(data_to_plot) != 1:
        n_tiles = len(data_to_plot)
    elif isinstance(grid_size, int):
        n_tiles = grid_size**2
    else:
        raise ValueError("'grid_size' must either be a single integer or a list of two integers.")

    if not isinstance(overlap, bool):
        raise ValueError("'overlap' must be 'True' or 'False'.")

    if not (
        isinstance(cmap, matplotlib.colors.Colormap)
        or (isinstance(cmap, list) and all(isinstance(c, matplotlib.colors.Colormap) for c in cmap))
    ):
        raise ValueError("'cmap' must either be a single or a list of matplotlib.colors.Colormap.")
    if (isinstance(cmap, list) and len(cmap) > 1) and not (len(cmap) == len(data_to_plot.keys())):
        raise ValueError(
            "'cmap' must either be length one or the same length as the channels that will be plotted."
        )

    if n_tiles > upper_limit_tiles:
        warnings.warn(
            "The generated plot will be very large and might slow down your machine. Consider plotting it outside of spatialmuon."
        )

    if len(data_to_plot) > n_tiles:
        msg = "More channels available than covered by 'grid_size'. Only the first {} channels will be plotted".format(n_tiles)
        warnings.warn(msg)
    
    if isinstance(cmap, matplotlib.colors.Colormap) and len(data_to_plot.keys()) > 1:
        cmap = default_cmaps
            
    if overlap == False:    
        
        # Calcualte grid layout
        if not isinstance(grid_size, list):
            n_x = math.ceil(n_tiles**0.5)
            n_y = math.floor(n_tiles**0.5)
            if n_x*n_y < n_tiles:
                n_y += 1 
        else:
            n_x = grid_size[0]
            n_y = grid_size[1]

        fig, axs = plt.subplots(n_y, n_x)
        
        if len(data_to_plot) > 1:
            axs = axs.flatten()

        for idx, channel in enumerate(data_to_plot):
            if idx < n_tiles:
                x = data_to_plot[channel] if preprocessing is None else preprocessing(data_to_plot[channel])
                if len(data_to_plot) > 1:
                    axs[idx].matshow(x, cmap=cmap[idx])
                    axs[idx].text(0, -10, channel, size=12)
                    for ax in axs.flat:
                        ax.set_axis_off()
                else:
                    axs.matshow(x, cmap=cmap)
                    axs.set_title(channel)
                    axs.set_axis_off()
                    
    elif overlap == True:
        
        fig, axs = plt.subplots(1, 1)
        
        for idx, channel in enumerate(data_to_plot):
            a = 1/(len(data_to_plot.keys())-1) if idx > 0 else 1
            x = data_to_plot[channel] if preprocessing is None else preprocessing(data_to_plot[channel])
            axs.matshow(x, cmap=cmap[idx], alpha=a)
        title = "background: {}; overlay: {}".format(
            [k for k in data_to_plot.keys()][0], 
            ", ".join(map(str, [k for k in data_to_plot.keys()][1:]))
        )
        
        legend = []
        for idx, c in enumerate(cmap):
            rgba = c(0.5)
            legend.append(
                matplotlib.patches.Patch(
                    facecolor=rgba, 
                    edgecolor=rgba,
                    label=[k for k in data_to_plot.keys()][idx]
                )
            )
            
        axs.legend(
            handles=legend, 
            frameon=False,
            loc="center left", 
            bbox_to_anchor=(1, 0.5)
        )
        axs.set_title(title)
        axs.set_axis_off()

    fig.tight_layout()
    fig.show()

    plt.style.use("default")