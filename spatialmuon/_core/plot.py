import math

from spatialmuon._core.fieldofview import FieldOfView
from spatialmuon._core.spatialmodality import SpatialModality
from typing import Union, Optional, Callable
import spatialmuon.datatypes
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.cm
import matplotlib.image

# import matplotlib.font_manager as fm
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np


def spatial(
    sm: SpatialModality,
    ax: matplotlib.axes.Axes,
    channel: Optional[Union[str, int]] = None,
    transform: Callable = None,
    scalebar=True,
    **kwargs,
):
    if len(sm) == 0:
        print("modality does not contain field of views")
        return
    elif len(sm) > 1:
        print("currently only modalities with 1 field of views can be plotted")
        return
    else:
        fov = sm.values().__iter__().__next__()
    if type(fov) == spatialmuon.datatypes.raster.Raster:
        spatial_raster(fov, ax, channel, transform, **kwargs)
    else:
        print("aaa")

    if scalebar:
        # fig = ax.get_figure()
        # dpi = fig.get_dpi()
        # fig_w, fig_h = fig.get_size_inches()
        # _, _, ax_w, ax_h = ax.get_position().bounds
        #
        # pixels_w = fig_w * ax_w * dpi
        # pixels_h = fig_h * ax_h * dpi
        # units_w = ax.get_xlim()[1] - ax.get_xlim()[0]
        # units_h = ax.get_ylim()[0] - ax.get_ylim()[1]
        # units_per_pixel_w = units_w / pixels_w
        # units_per_pixel_h = units_h / pixels_h
        # assert np.isclose(units_per_pixel_w, units_per_pixel_h)

        scalebar = ScaleBar(sm.scale, sm.coordinate_unit, box_alpha=0.8)
        ax.add_artist(scalebar)

        # print(width, height)
        # l = max(width, height) / 5
        # h = l / 20
        # # sm.scale
        # fontprops = fm.FontProperties(size=l * 18 / 20)
        # scalebar = AnchoredSizeBar(
        #     ax.transData,
        #     l,
        #     f"{l} {sm.coordinate_unit}",
        #     "lower center",
        #     pad=l / 5,
        #     color="red",
        #     frameon=False,
        #     size_vertical=h,
        #     fontproperties=fontprops,
        # )
        #
        # ax.add_artist(scalebar)


def spatial_raster(
    fov: FieldOfView,
    ax: matplotlib.axes.Axes,
    channel: Optional[Union[str, int]],
    transform: Callable,
    **kwargs,
) -> Optional[Union[matplotlib.image.AxesImage]]:
    if channel is not None:
        if type(channel) == str:
            # inefficient for many repeated calls
            if channel not in fov.channel_names:
                print("channel not found")
                return
            else:
                i = np.where(fov.channel_names)[0].item()
        elif type(channel) == int:
            i = channel
        else:
            raise ValueError()
        x = fov.X[...][:, :, i]
        if transform is not None:
            x = transform(x)
        im = ax.imshow(x, cmap=matplotlib.cm.get_cmap("gray"), **kwargs)
        return im
    else:
        print("channel is None")
        return
