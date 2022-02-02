from typing import Optional, Union, Callable, Dict
import warnings
import matplotlib
import matplotlib.cm
from spatialmuon._core.backing import BackableObject
from spatialmuon._core.fieldofview import FieldOfView, UnknownEncodingException
from spatialmuon.utils import _read_hdf5_attribute, _get_hdf5_attribute

import h5py


class SpatialModality(BackableObject):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        fovs: Optional[dict] = None,
    ):
        super().__init__(backing)
        if self.is_backed:
            for f, fov in self.backing.items():
                try:
                    self[f] = FieldOfView(backing=fov)
                except UnknownEncodingException as e:
                    warnings.warn(f"Unknown field of view type {e.encoding}")
        else:
            if fovs is not None:
                self.update(fovs)

    @staticmethod
    def _encodingtype():
        return "spatialmodality"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    # @property
    # def _backed_children(self) -> Dict[str, "BackableObject"]:
    #     d = {}
    #     for f, fov in self.items():
    #         d[f] = fov
    #     return d

    # TODO: the else branch of _set_backing has no equivalent in _backed_childer, check that it is all right/when
    #  that is called in a previous commit
    # def _set_backing(self, grp: Optional[h5py.Group] = None):
    #     super()._set_backing(grp)
    #     if grp is not None:
    #         self._write_attributes(grp)
    #         for f, fov in self.items():
    #             fov.set_backing(grp, f)
    #     else:
    #         for fov in self.values():
    #             fov.set_backing(None)

    def _write_attributes_impl(self, grp: h5py.Group):
        pass

    def _write_impl(self, obj: Union[h5py.Group, h5py.Dataset]):
        pass

    # def _write_impl(self, grp):
    #     for f, fov in self.items():
    #         fov._write(grp, f)

    # flake8: noqa: C901
    def plot(
        self,
        channels: Optional[Union[str, list[str]]] = "all",
        grid_size: Union[int, list[int]] = 1,
        preprocessing: Optional[Callable] = None,
        overlap: bool = False,
        cmap: Union[
            matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]
        ] = matplotlib.cm.viridis,
    ):

        if not (isinstance(channels, list) or isinstance(channels, str)):
            raise ValueError(
                "'channels' must be either a single character string or a list of them."
            )

        if isinstance(channels, list) and not all(isinstance(x, str) for x in channels):
            raise ValueError("If 'channels' is a list, all elements must be character strings.")

        valid_channels = {}  # will be used for more informative error messages
        for key, ome in self.data.items():
            valid_channels[key] = []
            for c in ome.var["channel_name"].tolist():
                valid_channels[key].append(key + "/" + c)
        valid_channels_flat = [k for k in valid_channels.keys()] + [
            x for v in valid_channels.values() for x in v
        ]

        if isinstance(channels, list):
            for c in channels:
                if c not in valid_channels_flat:
                    raise ValueError(
                        "'{}' not found in channels, available are: {}".format(
                            c, ", ".join(valid_channels_flat)
                        )
                    )

        if isinstance(channels, str) and channels != "all":
            if channels not in valid_channels_flat:
                raise ValueError(
                    "'{}' not found in channels, available are: {}".format(
                        channels, ", ".join(valid_channels_flat)
                    )
                )

        if channels == "all":
            channels_to_plot = [x for v in valid_channels.values() for x in v]

        if isinstance(channels, str) and channels not in [k for k in self.data.keys()]:
            channels_to_plot = channels
        elif isinstance(channels, str) and channels in [k for k in self.data.keys()]:
            channels_to_plot = [
                channels + "/" + c for c in self.data[channels].var["channel_name"].tolist()
            ]

        if isinstance(channels, list):
            # Split user input list into full FOVs (which need to be dissected) and individual channels, will later be reunited
            full_fovs = []
            partial_fov_channels = []
            channels_to_plot = []
            for c in channels:
                if c in [k for k in self.data.keys()]:
                    full_fovs.append(c)
                else:
                    partial_fov_channels.append(c)
            if len(full_fovs) > 0:
                for ff in full_fovs:
                    channels_to_plot = channels_to_plot + [
                        ff + "/" + c for c in self.data[ff].var["channel_name"].tolist()
                    ]
            channels_to_plot = channels_to_plot + partial_fov_channels

        data_to_plot = {}  # will be passed to the plotting function
        for c in channels_to_plot:

            fov, channel = c.split("/")
            channel_idx = (
                self.data[fov].var.query("channel_name == '{}'".format(channel)).index.tolist()[0]
            )
            data_to_plot[c] = self.data[fov].X[:, :, channel_idx]

        plot_preview_grid(
            data_to_plot=data_to_plot,
            grid_size=grid_size,
            preprocessing=preprocessing,
            overlap=overlap,
            cmap=cmap,
        )
