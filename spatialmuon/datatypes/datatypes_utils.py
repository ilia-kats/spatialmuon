from typing import Optional, Union, Callable

import matplotlib
import matplotlib.cm
import matplotlib.axes
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt


def regions_raster_plot(
    instance,
    channels: Optional[Union[str, list[str]]] = "all",
    grid_size: Union[int, list[int]] = 1,
    preprocessing: Optional[Callable] = None,
    overlap: bool = False,
    cmap: Union[
        matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]
    ] = matplotlib.cm.viridis,
    ax: matplotlib.axes.Axes = None,
):
    if not (isinstance(channels, list) or isinstance(channels, str)):
        raise ValueError("'channels' must be either a single character string or a list of them.")

    if isinstance(channels, list) and not all(isinstance(x, str) for x in channels):
        raise ValueError("If 'channels' is a list, all elements must be character strings.")

    if isinstance(channels, list):
        for c in channels:
            if c not in instance.var["channel_name"].tolist():
                raise ValueError(
                    "'{}' not found in channels, available are: {}".format(
                        c, ", ".join(map(str, instance.var["channel_name"]))
                    )
                )

    if isinstance(channels, str) and channels != "all":
        if channels not in instance.var["channel_name"].tolist():
            raise ValueError(
                "'{}' not found in channels, available are: {}".format(
                    channels, ", ".join(map(str, instance.var["channel_name"]))
                )
            )
    if channels == "all":
        channels_to_plot = instance.var["channel_name"].tolist()
    else:
        channels_to_plot = channels if isinstance(channels, list) else [channels]

    if not (
        isinstance(cmap, matplotlib.colors.Colormap)
        or (isinstance(cmap, list) and all(isinstance(c, matplotlib.colors.Colormap) for c in cmap))
    ):
        raise ValueError("'cmap' must either be a single or a list of matplotlib.colors.Colormap.")
    if (isinstance(cmap, list) and len(cmap) > 1) and not (len(cmap) == len(channels_to_plot)):
        raise ValueError(
            "'cmap' must either be length one or the same length as the channels that will be plotted."
        )

    if not isinstance(overlap, bool):
        raise ValueError("'overlap' must be 'True' or 'False'.")

    if ax is not None:
        overlap = True

    if overlap or len(channels_to_plot) == 1:
        instance._plot_in_canvas(
            channels_to_plot=channels_to_plot, preprocessing=preprocessing, cmap=cmap, ax=ax
        )
    else:
        instance._plot_in_grid(
            channels_to_plot=channels_to_plot,
            grid_size=grid_size,
            preprocessing=preprocessing,
            cmap=cmap,
        )
