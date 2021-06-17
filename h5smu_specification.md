# [RFC][v0.2.1] spatial muon HDF5 storage spec
## file format
use HDF5's user block feature to write a custom header of the form `SpatialMuData (format-version=0.1.0;creator=package_name;creator-version=package_version)`. This will make the file immediately identifiable as a SpatialMuon file with any file extension and without having HDF5 installed.

## top-level
- **attributes**
    - `encoder`: String, writing library
    - `encoder-version`: String, version of the writing library
    - `encoding`: `SpatialMuData`
    - `encoding-version`: String, version of the format

## group `mod`
contains one sub-group per modality

## `mod/`modality subgroup
- **data sets**
    - `scale`: Optional. Float, scale factor of this modality. If missing, different modalities cannot be aligned.
    - `coordinate_unit`: Optional. String, unit of the coordinates for this modality (e.g. `µm`).

        the assumption is that all FOVs in a modality have the same scale factor
- **groups**

    one subgroup per field of view

## `mod/modality/`FOV subgroup
- **attributes**
    - `encoding`: One of
        - `array` for array data, e.g. Visium/SlideSeq
        - `single-molecule` for single-molecule data (e.g. SeqFISH)
        - `raster` for IMC data, where expression data are on a regular grid and are stored as images
    - `encoding-version`: version of the format

    list of encodings can be extended in the future

- **data sets**
    - `rotation`: rotation matrix to align this FOV with the other FOVs. Can be 2D or 3D.
    - `translation`: 2 or 3 element translation vector to align this FOV with the other FOVs.
    - additional data sets depending on type (array/single-molecule), see below

- **groups**
    - `index` spatial index for fast subsetting of points by masks or neighbors (I have hacked together a preliminary version of this using [`rtree`](https://github.com/Toblerity/rtree/))
    - `images` contains images in different resolution
    - `feature_masks`: contains masks for selecting features based on spatial location. See below for storage.
    - `image_masks`: contains masks for selecting from images. Contains one group per image resolution, masks are stored in the same way as `raster` feature masks.
    - `uns`: unstructured metadata

## `mod/modality/FOV/feature_masks` subgroup
Each member of this group has the following attributes:

- `encoding` one of `polygon`, `raster`
- `encoding-version`: version of the format

### polygon masks
This is the only supported format for single-molecule and Visium data. Each mask is a group, allowing one to store sets of masks under the same name (e.g. all cells from a particular processing steps). Each mask is an n_vertices x n_dim array, open (i.e. the last vertex is distinct from the first vertex and the polygon will be connected automatically).

### raster masks
Only supported for IMC data at the moment. Stored as 2D or 3D images (depending on how many dimensions the data has) with integer values. `0` encodes background (no selection), different values represent different objects.

## `mod/modality/FOV/images` subgroup
contains one group per resolution. Each resolution group contains one group per image, containing:

- **data sets**
    - `scale`: float, scale factor of this image to align it with the measurement coordinates
    - `px_width`: width of one pixel in the same units that the coordinates are in
    - `px_height`: height of one pixel in the same units that the coordinates are in
    - `rotation`: 2D/3D rotaion matrix to align measurement coordinates with the image.
    - `translation`: 2 or 3-element translation vector to align measurement coordinates with the image.
    - `channel_names`: vector of strings identifying the channels. Optional for single-channel or 3-channel images. 3-channel images with missing channel names will be assumed to be RGB.
    - `image`: the actual image.

For example, `mod/visium/slice1/images/50000x50000/HnE/` would be a group

## `mod/modality/FOV` array data
- **additional data sets**
    - `coordinates`: n_obs x n_dim array of spot coordinates
    - `X`: dense or sparse n_obs x n_genes matrix
    - `spot_shape`: string, one of `circle`, `square`
    - `spot_size`: size of a spot. If `spot_shape` is circle, scalar value giving the radius. If `spot_shape` is square, 1d array of length 2 or 3, size of a spot in each dimension

- ** additional groups**
    - `var`: data frame (encoded using AnnData's format) with per-gene metadata
    - `obs`: data frame with per-spot metadata

Do we need `obsm`/`varm`?

## `mod/modality/FOV` single-molecule data
coordinates will be stored sorted by feature name, which will allow efficient subsetting by feature name using the `feature_ranmge` group.

- **additional data sets**
    - `coordinates`: n_obs x n_dim array of molecule coordinates
    - `feature_name`: string array with n_obs elements. Would typically be used for gene names

- **additional groups**
    - `metadata`: data frame (encoded using AnnData's format) with per-molecule metadata
    - `feature_range`: group, contains one data set per unique feature name. The data set is a 2-element array containing the first and last indices (0-indexed, exclusive) of coordinates corresponding to the respective feature. This allows for fast subsetting by feature name: For in-memory storage, the entire group can be read into a dict. For backed storage, HDF5 stores groups as a B-tree, so lookup should happen in logarithmic time.

Do we need something like the `.obs` dataframe for single-molecule data?

## `mod/modality/FOV` raster data
- **additional data sets**
    - `X`: 3- or 4-dimensional image. Last dimension is channels, contains one channel per feature.
    - `px_distance`: 1d array of length 2 or 3, center-to-center distance between pixels in each dimension
    - `px_size`: 1d array of length 2 or 3, size of a pixel in each dimension (useful in case there is inter-pixel distance)
