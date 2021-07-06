from typing import Optional

class SpatialModality:
    def __init__(self, fovs:dict = {}, scale: Optional[float]=None, coordinate_unit: Optional[str]=None):
        self.fovs = fovs
        self._scale = scale
        self.coordinate_unit = coordinate_unit

    @property
    def scale(self):
        return self._scale if self._scale is not None and self._scale > 0 else 1

    @scale.setter
    def scale(self, newscale:Optional[float]):
        if newscale is not None and newscale <= 0:
            newscale = None
        self._scale = newscale

    def __getitem__(self, key):
        return self.fovs[key]
