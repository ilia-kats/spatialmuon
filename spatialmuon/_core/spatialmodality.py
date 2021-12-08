from typing import Optional, Union, Callable
import warnings

from .backing import BackableObject, BackedDictProxy
from .fieldofview import FieldOfView, UnknownEncodingException
from ..utils import _read_hdf5_attribute, _get_hdf5_attribute

import h5py


class SpatialModality(BackableObject, BackedDictProxy):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        fovs: Optional[dict] = None,
        coordinate_unit: Optional[str] = None,
    ):
        super().__init__(backing)
        if self.isbacked:
            for f, fov in self.backing.items():
                try:
                    self[f] = FieldOfView(backing=fov)
                except UnknownEncodingException as e:
                    warnings.warn(f"Unknown field of view type {e.encoding}")
            self.coordinate_unit = _get_hdf5_attribute(self.backing.attrs, "coordinate_unit", None)
        else:
            if fovs is not None:
                self.update(fovs)
            self.coordinate_unit = coordinate_unit

    @staticmethod
    def _encodingtype():
        return "spatialmodality"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, grp: Optional[h5py.Group] = None):
        super()._set_backing(grp)
        if grp is not None:
            self._write_attributes(grp)
            for f, fov in self.items():
                fov.set_backing(grp, f)
        else:
            for fov in self.values():
                fov.set_backing(None)

    def _write_attributes_impl(self, grp: h5py.Group):
        if self.coordinate_unit is not None:
            grp.attrs["coordinate_unit"] = self.coordinate_unit

    def _write(self, grp):
        for f, fov in self.items():
            fov.write(grp, f)
            
    def plot(
          self, 
          channels: Optional[Union[str, list[str]]] = "all",
          grid_size: Union[int, list[int]] = 1,
          preprocessing: Optional[Callable] = None
        ):
        
        if not (isinstance(channels, list) or isinstance(channels, str)):
            raise ValueError("'channels' must be either a single character string or a list of them.")

        if isinstance(channels, list) and not all(isinstance(x, str) for x in channels):
            raise ValueError("If 'channels' is a list, all elements must be character strings.")

        valid_channels = {}
        for key, ome in self.data.items():
            valid_channels[key] = []
            for c in ome.var["channel_name"].tolist():
                valid_channels[key].append(key + "/" + c)
        valid_channels_flat = [k for k in valid_channels.keys()] + [x for v in valid_channels.values() for x in v]
        
        if isinstance(channels, list):
            for c in channels:
                if c not in valid_channels_flat:
                    raise ValueError("'{}' not found in channels, available are: {}".format(c, ', '.join(valid_channels_flat)))

        if isinstance(channels, str) and channels != "all":
            if channels not in valid_channels_flat:
                raise ValueError("'{}' not found in channels, available are: {}".format(channels, ', '.join(valid_channels_flat)))
                
        if channels == "all":
            channels_to_plot = [x for v in valid_channels.values() for x in v]
            
        if isinstance(channels, str) and channels not in [k for k in self.data.keys()]:
            channels_to_plot = channels
        elif isinstance(channels, str) and channels in [k for k in self.data.keys()]:
            channels_to_plot = [channels + "/" + c for c in self.data[channels].var["channel_name"].tolist()]
        
        if isinstance(channels, list):
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
                    channels_to_plot = channels_to_plot + [ff + "/" + c for c in self.data[ff].var["channel_name"].tolist()]
            channels_to_plot = channels_to_plot + partial_fov_channels
            
        print(channels_to_plot)
            
