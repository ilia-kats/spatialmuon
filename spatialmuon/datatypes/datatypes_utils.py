from typing import Optional, Union, Callable

import matplotlib
import matplotlib.cm
import matplotlib.axes
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import warnings
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_channel_index_from_channel_name(var, channel_name):
    channel_idx = var.query("channel_name == '{}'".format(channel_name)).index.tolist()[0]
    return channel_idx


# flake8: noqa: C901
def regions_raster_plot(
    instance,
    channels: Optional[Union[str, list[str], int, list[int]]] = "all",
    grid_size: Union[int, list[int]] = 1,
    preprocessing: Optional[Callable] = None,
    overlap: bool = False,
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

    if not isinstance(overlap, bool):
        raise ValueError("'overlap' must be 'True' or 'False'.")

    if ax is not None:
        overlap = True

    if overlap or len(channels_to_plot) == 1:
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
            channels_to_plot=channels_to_plot, preprocessing=preprocessing, cmap=cmap, ax=axs
        )
        if show_title:
            title = "background: " if len(channels_to_plot) > 1 else ""
            title += "{}".format([k for k in channels_to_plot][0])
            if len(channels_to_plot) > 1:
                title += "; overlay: {}".format(
                    ", ".join(map(str, [k for k in channels_to_plot][1:]))
                )
                axs.set_title(title)
        if show_legend:
            _legend = []
            for idx, c in enumerate(cmap):
                rgba = c(255)
                _legend.append(
                    matplotlib.patches.Patch(
                        facecolor=rgba, edgecolor=rgba, label=[k for k in channels_to_plot][idx]
                    )
                )

            axs.legend(
                handles=_legend,
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.1),
                ncol=len(_legend),
            )
        if len(channels_to_plot) == 1 and show_colorbar:
            # divider = make_axes_locatable(axs)
            # cax = divider.append_axes("bottom", size="5%", pad=0.05)
            plt.colorbar(
                im, orientation="horizontal", location="bottom", ax=axs, shrink=0.6, pad=0.055
            )
        if show_scalebar:
            unit = instance.coordinate_unit
            if unit is not None:
                scalebar = ScaleBar(
                    instance.scale, unit, box_alpha=0.8, color="white", box_color="black"
                )
                axs.add_artist(scalebar)

        # axs[idx].text(0, -10, channel, size=12)
        if suptitle is not None:
            plt.suptitle(suptitle)
        axs.set_axis_off()
        if ax is None:
            plt.tight_layout()
            plt.show()
    else:
        # ######### going back to the calling class ########## #
        instance._plot_in_grid(
            channels_to_plot=channels_to_plot,
            grid_size=grid_size,
            preprocessing=preprocessing,
            cmap=cmap,
            suptitle=suptitle,
        )
