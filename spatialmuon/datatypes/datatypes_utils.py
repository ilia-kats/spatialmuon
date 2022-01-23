from typing import Optional, Union, Callable, Literal

import matplotlib
import matplotlib.cm
import matplotlib.axes
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import warnings
import math
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

import spatialmuon

PlottingMethod = Literal["auto", "panels", "overlap", "rgba"]


def get_channel_index_from_channel_name(var, channel_name):
    channel_idx = var.query("channel_name == @channel_name").index.tolist()[0]
    return channel_idx


# flake8: noqa: C901
def regions_raster_plot(
    instance,
    channels: Optional[Union[str, list[str], int, list[int]]] = "all",
    grid_size: Union[int, list[int]] = 1,
    preprocessing: Optional[Callable] = None,
    method: PlottingMethod = "auto",
    cmap: Union[
        matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]
    ] = matplotlib.cm.viridis,
    ax: matplotlib.axes.Axes = None,
    show_title: bool = True,
    show_legend: bool = True,
    show_colorbar: bool = True,
    show_scalebar: bool = True,
    suptitle: Optional[str] = None,
):
    if not (isinstance(channels, list) or isinstance(channels, str) or isinstance(channels, int)):
        raise ValueError(
            "'channels' must be either a single character string, an integer or a list of them."
        )

    if isinstance(channels, list) and not all(
        isinstance(x, str) or isinstance(x, int) for x in channels
    ):
        raise ValueError(
            "If 'channels' is a list, all elements must be either character strings or integers."
        )

    if isinstance(channels, list):
        for c in channels:
            if not isinstance(c, int) and c not in instance.var["channel_name"].tolist():
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
        if isinstance(channels, int):
            channels_to_plot = [instance.var["channel_name"][channels]]
        elif isinstance(channels, str):
            channels_to_plot = [channels]
        else:
            assert isinstance(channels, list)
            channels_to_plot = []
            for c in channels:
                if isinstance(c, int):
                    channels_to_plot.append(instance.var["channel_name"][c])
                else:
                    assert isinstance(c, str)
                    channels_to_plot.append(c)

    if not (
        isinstance(cmap, matplotlib.colors.Colormap)
        or (isinstance(cmap, list) and all(isinstance(c, matplotlib.colors.Colormap) for c in cmap))
    ):
        raise ValueError("'cmap' must either be a single or a list of matplotlib.colors.Colormap.")
    if (isinstance(cmap, list) and len(cmap) > 1) and not (len(cmap) == len(channels_to_plot)):
        raise ValueError(
            "'cmap' must either be length one or the same length as the channels that will be plotted."
        )

    if not method in ["auto", "panels", "overlap", "rgba"]:
        raise ValueError("plotting 'method' not recognized")

    if ax is not None:
        assert method != "panels"

    if len(channels_to_plot) == 1:
        method = "overlap"

    if method == "auto":
        if len(channels_to_plot) <= 4 and isinstance(instance, spatialmuon.Raster):
            method = "rgba"
        else:
            method = "panels"

    if method in ["overlap", "rgba"]:
        DEFAULT_CMAPS = [
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
        if method == "overlap":
            if isinstance(cmap, matplotlib.colors.Colormap) and len(channels_to_plot) > 1:
                cmap = DEFAULT_CMAPS
            if len(channels_to_plot) > 1:
                if len(channels_to_plot) > len(cmap):
                    warnings.warn(
                        f"{len(channels_to_plot)} channels to plot but {len(cmap)} colormaps available; colormaps will be cycled."
                    )
                    cmap = cmap * len(channels_to_plot) // len(cmap) + 1
            if isinstance(cmap, matplotlib.colors.Colormap):
                cmap = [cmap]
            cmap = cmap[: len(channels_to_plot)]

        if len(channels_to_plot) == 1:
            show_legend = False and show_legend
            show_colorbar = True and show_colorbar
        else:
            show_legend = True and show_legend
            show_colorbar = False and show_colorbar

        if ax is None:
            fig, axs = plt.subplots(1, 1)
        else:
            axs = ax
        # ######### going back to the calling class ########## #
        im = instance._plot_in_canvas(
            channels_to_plot=channels_to_plot,
            rgba=method == "rgba",
            preprocessing=preprocessing,
            cmap=cmap,
            ax=axs,
        )
        if show_title:
            if method == "overlap":
                title = "background: " if len(channels_to_plot) > 1 else ""
                title += "{}".format([k for k in channels_to_plot][0])
                if len(channels_to_plot) > 1:
                    title += "; overlay: {}".format(
                        ", ".join(map(str, [k for k in channels_to_plot][1:]))
                    )
            elif method == "rgba":
                title = f'RGB: {", ".join(map(str, [k for k in channels_to_plot][:3]))}'
                if len(channels_to_plot) == 4:
                    title += f". Alpha: {[k for k in channels_to_plot][-1]}"
            else:
                raise ValueError()
            axs.set_title(title)
        if show_legend and not (method == 'rgba' and isinstance(instance, spatialmuon.Raster)):
            _legend = []
            if method == "overlap":
                for idx, c in enumerate(cmap):
                    rgba = c(255)
                    # label = itertools.islice(channels_to_plot, idx, None).__iter__().__next__()
                    # label = channels_to_plot[idx]
                    label = [k for k in channels_to_plot][idx]
                    _legend.append(
                        matplotlib.patches.Patch(facecolor=rgba, edgecolor=rgba, label=label)
                    )
            elif method == "rgba":
                colors = ["red", "green", "blue", "white"]
                for idx, label in enumerate(channels_to_plot):
                    c = colors[idx]
                    if idx == 3:
                        label = "(alpha) " + label
                    _legend.append(
                        matplotlib.patches.Patch(facecolor=c, edgecolor="k", label=label)
                    )
            else:
                raise ValueError()

            axs.legend(
                handles=_legend,
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=len(_legend),
            )
        if len(channels_to_plot) == 1 and show_colorbar:
            # divider = make_axes_locatable(axs)
            # cax = divider.append_axes("bottom", size="5%", pad=0.05)
            plt.colorbar(
                im, orientation="horizontal", location="bottom", ax=axs, shrink=0.6, pad=0.1
            )
        if show_scalebar:
            unit = instance.coordinate_unit
            if unit is not None:
                scale_factor = np.linalg.norm(instance.anchor.vector)
                try:
                    scalebar = ScaleBar(
                        scale_factor, unit, box_alpha=0.8, color="white", box_color="black"
                    )
                except ValueError as e:
                    if str(e).startswith("Invalid unit (") and str(e).endswith(") with dimension"):
                        from matplotlib_scalebar.dimension import _Dimension

                        class CustomDimension(_Dimension):
                            def __init__(self, unit: str):
                                super().__init__(unit)

                        custom_dimension = CustomDimension(unit=unit)
                        ##
                        scalebar = ScaleBar(
                            scale_factor,
                            units=unit,
                            dimension=custom_dimension,
                            box_alpha=0.8,
                            color="white",
                            box_color="black",
                        )
                        ##
                    else:
                        raise e
                axs.add_artist(scalebar)

        instance._adjust_plot_lims(axs)
        # axs[idx].text(0, -10, channel, size=12)
        if suptitle is not None:
            plt.suptitle(suptitle)
        # axs.set_axis_off()
        if ax is None:
            plt.tight_layout()
            plt.show()
    else:
        assert method == "panels"
        upper_limit_tiles = 100

        if (
            isinstance(grid_size, list)
            and len(grid_size) == 2
            and all(isinstance(x, int) for x in grid_size)
        ):
            n_tiles = grid_size[0] * grid_size[1]
        elif grid_size == 1 and len(channels_to_plot) != 1:
            n_tiles = len(channels_to_plot)
        elif isinstance(grid_size, int):
            n_tiles = grid_size ** 2
        else:
            raise ValueError(
                "'grid_size' must either be a single integer or a list of two integers."
            )
        if n_tiles > upper_limit_tiles:
            raise RuntimeError(
                f"Not plotting more than {upper_limit_tiles} subplots. Please plot channels to Axes manually"
            )

        if len(channels_to_plot) > n_tiles:
            msg = "More channels available than covered by 'grid_size'. Only the first {} channels will be plotted".format(
                n_tiles
            )
            warnings.warn(msg)

        # Calcualte grid layout
        if not isinstance(grid_size, list):
            n_x = math.ceil(n_tiles ** 0.5)
            n_y = math.floor(n_tiles ** 0.5)
            if n_x * n_y < n_tiles:
                n_y += 1
        else:
            n_x = grid_size[0]
            n_y = grid_size[1]
        grid_size = [n_x, n_y]

        # ######### going back to the calling class ########## #
        instance._plot_in_grid(
            channels_to_plot=channels_to_plot,
            grid_size=grid_size,
            preprocessing=preprocessing,
            cmap=cmap,
            suptitle=suptitle,
        )
