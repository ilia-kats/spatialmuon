# [RFC][v0.3.0] spatial muon HDF5 storage spec
## file format
use HDF5's user block feature to write a custom header of the form `SpatialMuData (format-version=0.1.0;creator=package_name;creator-version=package_version)`. This will make the file immediately identifiable as a SpatialMuon file with any file extension and without having HDF5 installed.

## top-level
- **attributes**
    - `encoder`: String, writing library
    - `encoder-version`: String, version of the writing library
    - `encoding-type`: `SpatialMuData`
    - `encoding-version`: String, version of the format

## group `mod`
contains one sub-group per modality

## `mod/`modality subgroup
- **attributes**
    - `encoding-type`: `spatialmodality`
    - `encoding-version`: 0.1.0
    - `coordinate_unit`: Optional. String, unit of the coordinates for this modality (e.g. `µm`).

    the assumption is that all FOVs in a modality have the same scale factor
- **groups**

    one subgroup per field of view

## `mod/modality/`FOV subgroup
- **attributes**
    - `encoding-type`: One of
        - `fov-array` for array data, e.g. Visium/SlideSeq
        - `fov-single-molecule` for single-molecule data (e.g. SeqFISH)
        - `fov-raster` for IMC data, where expression data are on a regular grid and are stored as images
    - `encoding-version`: version of the format
    - `scale`: Positive float, scale factor of this modality, default to 1.
    - `rotation`: rotation matrix to align this FOV with the other FOVs. Can be 2D or 3D. Optional, if missing assumed to be the identity matrix. It needs to have determinant with module 1.
    - `center_of_rotation`: , reference point for rotation, default (0, 0, ...), primarily used for spm.Raster()
    - `translation`: 2 or 3 element translation vector to align this FOV with the other FOVs.
    - additional data sets depending on type (array/single-molecule), see below

    list of encodings can be extended in the future

- **groups**
    - `images` contains images in different resolution
    - `masks`: contains masks for definiting regions in space (for instance they could be used for selecting features based on spatial location). See below for storage.
    - `var`: data frame (encoded using AnnData's format) with per-gene/per-feature metadata
    - `uns`: unstructured metadata

## `mod/modality/FOV/masks` subgroup
Each member of this group has the following attributes:

- `encoding-type` one of `mask-polygon`, `mask-mesh`, `mask-raster`
- `encoding-version`: version of the format

Also an additional group containing information regarding the masks:
- `obs`: data frame with per-spot metadata

### shape masks
Supported for 2D and 3D single-molecule, regions and raster data.
Masks defined by simple geometrics primitives which allow for efficient storage and operations. This will probably be the most common type of masks that the user will be interested in.

Example of encoding of circular shapes:
    - `spot_shape`: string, one of `circle`, `rectangle`
    - `spot_size`: size of a spot. If `spot_shape` is circle, scalar value giving the radius. If `spot_shape` is rectangle, 1d array of length 2 or 3, size of a spot in each dimension

### polygon masks
Supported for 2D and 3D single-molecule, regions and raster data. Each mask is an n_vertices x n_dim array, open (i.e. the last vertex is distinct from the first vertex and the polygon will be connected automatically). The mask is a group and contains the list of the coordinates of the vertices.

Polygons are inherently 2-dimensional, even if embedded into a 3D space. Therefore, if the data set is 3D, subsetting by a polygon mask is defined by default as first projecting all coordinates onto their first 2 principal components, and then applying the mask. This is equivalent to projecting the polygon into the data coordinate system, which is how it's implemented in order to take advantage of the spatial index. This behavour can be disabled by the user, in which case the 3rd axis will simply be discarded when applying the mask.

### mesh masks
Supported for 3D single-molecule (and potentially array) data. Each mask is again a group with datasets:

- `vertices`: n x 3 array with coordinates of each vertex
- `faces`: m x 3 or m x 4 (for quad faces) array. Each entry is the index of the corresponding vertex in the vertices array

### raster masks
Only supported for IMC data at the moment. Stored as 2D or 3D images (depending on how many dimensions the data has) with integer values. `0` encodes background (no selection), different values represent different objects.

## `mod/modality/FOV` "regions" data
- **data sets**
    - `coordinates`: n_obs x n_dim array of spot coordinates
    - `X`: dense or sparse n_obs x n_genes matrix

- ** additional groups**
    - `index` spatial index for fast subsetting of points by masks or neighbors

Do we need `obsm`/`varm`?

## `mod/modality/FOV` single-molecule data
coordinates will be stored sorted by feature name, which will allow efficient subsetting by feature name using the `feature_range` group.

- **data sets**
    - `coordinates`: n_obs x n_dim array of molecule coordinates

- **additional groups**
    - `index` spatial index for fast subsetting of points by masks or neighbors
    - `metadata`: data frame (encoded using AnnData's format) with per-molecule metadata. Data frame index is the gene name.
    - `feature_range`: group, contains one data set per unique feature name. The data set is a 2-element array containing the first and last indices (0-indexed, exclusive) of coordinates corresponding to the respective feature. This allows for fast subsetting by feature name: For in-memory storage, the entire group can be read into a dict. For backed storage, HDF5 stores groups as a B-tree, so lookup should happen in logarithmic time.

Do we need something like the `.obs` dataframe for single-molecule data?

## `mod/modality/FOV` raster data
2d or 3d image with 1 or multiple channels. A regular color image will be stored in this format as a 2d image with 3 (or 4) channels (rgb, rgba).

- `base_resolution`: 2-element (or 3-element) integer vector containing width and height (and depth) of the original 2d (3d) image resolution to which the scale factor, translation vector, rotation matrix, pixel size and pixel-to-pixel distance refers to.
- **data sets**
    one data set for each resolution, containing the lower/higher resolution 2d/3d image

For example, `mod/visium/slice1/images/HnE/50000x50000` would be a data set
- **additional attributes**
    - `px_dimensions`: 1d array of length 2 or 3, size of a pixel in each dimension. Optional, if missing assumed to be 1.
    - `px_distance`: 1d array of length 2 or 3, center-to-center distance between pixels in each dimension (useful in case there is inter-pixel distance). Optional, if missing assumed to be `px_dimensions`.

- **data sets**
    - `X`: 2-, 3- or 4-dimensional image. Last dimension is channels, contains one channel per feature. 2-dimensional images are assumed to contain a single channel.
