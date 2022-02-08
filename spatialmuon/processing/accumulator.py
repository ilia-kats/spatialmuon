from functools import singledispatch
import numpy as np
import vigra
import copy

from spatialmuon._core.masks import Masks, RasterMasks
from spatialmuon.datatypes.raster import Raster
from spatialmuon.datatypes.regions import Regions
from spatialmuon.datatypes.singlemolecule import SingleMolecule
from spatialmuon._core.fieldofview import FieldOfView

def accumulate_features(masks: Masks, fov: FieldOfView) -> Regions:
    if isinstance(fov, Regions):
        return _accumulate_features_on_regions()
    elif isinstance(fov, Raster):
        return _accumulate_features_on_raster()
    elif isinstance(fov, SingleMolecule):
        return _accumulate_features_on_single_molecule()
    else:
        raise ValueError()

@singledispatch
def _accumulate_features_on_regions(masks: Masks, regions: Regions) -> Regions:
    raise NotImplementedError()

@singledispatch
def _accumulate_features_on_raster(masks: Masks, raster: Raster) -> Regions:
    raise NotImplementedError()

@_accumulate_features_on_raster.register
def _(masks: RasterMasks, raster: Raster) -> Regions:
    from spatialmuon.datatypes.regions import Regions
    x = raster.X[...]
    x = x.astype(np.float32)
    if len(x.shape) != 3:
        raise NotImplementedError('3D case')
    ome = np.require(x, requirements=["C"])
    vigra_ome = vigra.taggedView(ome, "xyc")
    masks = masks.X[...]
    masks = masks.astype(np.uint32)
    ##
    features = vigra.analysis.extractRegionFeatures(
        vigra_ome,
        labels=masks,
        ignoreLabel=0,
        features=["Count", "Maximum", "Mean", "Sum", "Variance", "RegionCenter"],
    )
    ##
    features = {k: v for k, v in features.items()}
    masks_with_obs = copy.copy(masks)
    original_labels = masks_with_obs.obs["original_labels"].to_numpy()
    if (
            "region_center_x" not in masks_with_obs.obs
            and "region_center_y" not in masks_with_obs.obs
    ):
        masks_with_obs.obs["region_center_x"] = features["RegionCenter"][original_labels, 1]
        masks_with_obs.obs["region_center_y"] = features["RegionCenter"][original_labels, 0]
    if "count" not in masks_with_obs.obs:
        masks_with_obs.obs["count"] = features["Count"][original_labels]
    d = {}
    for key in ["Maximum", "Mean", "Sum", "Variance"]:
        regions = Regions(
            backing=None,
            X=features[key][original_labels, :],
            var=raster.var,
            masks=copy.copy(masks_with_obs),
            coordinate_unit=raster.coordinate_unit,
        )
        d[key.lower()] = regions
    return d

@singledispatch
def _accumulate_features_on_single_molecule(masks: Masks, sm: SingleMolecule) -> Regions:
    raise NotImplementedError()