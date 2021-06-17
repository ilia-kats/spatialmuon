from tqdm import tqdm
import zipfile
import requests
import tifffile
import tempfile
import os
import math
import h5py
import hashlib
import argparse
import numpy as np

OME_AND_MASKS_URL = 'https://zenodo.org/record/3518284/files/OMEandSingleCellMasks.zip'
OME_AND_MASKS_HASH = '777f8a59da4f4efc2fcd7149565dd191'
METADATA_URL = 'https://zenodo.org/record/3518284/files/SingleCell_and_Metadata.zip'
METADATA_HASH = '157756ca703e6cfc73377c60d39dcb19'

parser = argparse.ArgumentParser(description='Download IMC data and makes it ready for spatial muon')
parser.add_argument('--dkfz', help='Use the DKFZ proxy for downloading', action='store_true', default=False)
args = parser.parse_args()

if args.dkfz:
    PROXIES = {
        'http': 'http://193.174.53.86:80',
        'https': 'https://193.174.53.86:80'
    }
else:
    PROXIES = None
CHUNK_SIZE = 8192


def download_file(url, output_dir):
    local_filename = os.path.join(output_dir, url.split('/')[-1])
    with requests.get(url, stream=True, proxies=PROXIES) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            bar = tqdm(total=int(r.headers['Content-Length']), desc='downloading')
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()
    return local_filename


def verify_file(path, hash):
    hash_md5 = hashlib.md5()
    file_size = os.stat(path).st_size
    bar = tqdm(total=file_size, desc='verifying hash')
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b''):
            hash_md5.update(chunk)
            bar.update(len(chunk))
    computed_hash = hash_md5.hexdigest()
    bar.close()
    assert hash == computed_hash


def unzip_file(f, force=False, remove_archive=False):
    assert f.endswith('.zip')
    out = f[:-4]
    if force or not os.path.isdir(out):
        with zipfile.ZipFile(f, 'r') as zf:
            for member in tqdm(zf.infolist(), desc='extracting'):
                # try:
                zf.extract(member, out)
                # except zipfile.error as e:
                #     pass
    if remove_archive and os.path.isfile(f):
        os.unlink(f)
    return out


def create_muon_spatial_object(f_ome, f_masks, outfile):
    ome = tifffile.imread(f_ome)
    masks = tifffile.imread(f_masks)
    with h5py.File(outfile, 'w', userblock_size=512):
        pass

    with open(outfile, 'rb+') as f:
        f.seek(0)
        header = 'SpatialMuData (format-version=0.2.1;creator=draft_luca;creator-version=draft_luca)'
        f.write(header.encode('utf-8'))

    with h5py.File(outfile, 'r+') as f5:
        # /
        # attributes
        f5.attrs['encoder'] = 'spatialmuon'
        f5.attrs['encoder-version'] = 'draft_luca'
        f5.attrs['encoding'] = 'SpatialMuData'
        f5.attrs['encoding-version'] = '0.2.1'

        # /mod/
        f5_mod = f5.create_group('mod')

        # /mod/imc/
        f5_imc = f5_mod.create_group('imc')
        # datasets
        f5_imc['scale'] = np.array([1], dtype=np.float)
        f5_imc['coordinate_unit'] = np.array('micrometre', dtype=h5py.string_dtype())

        # /mod/imc/fov
        f5_fov = f5_imc.create_group('FOV')
        # attributes
        f5_fov.attrs['encoding'] = 'raster'
        f5_fov.attrs['encoding-version'] = 'draft_luca'
        # datasets
        f5_fov['rotation'] = np.eye(2)
        f5_fov['translation'] = np.zeros((2,))
        f5_fov['X'] = ome.transpose((1, 2, 0))
        f5_fov['px_distance'] = np.array([1., 1.])
        f5_fov['px_size'] = np.array([1., 1.])

        # /mod/imc/fov/feature_masks
        f5_masks = f5_fov.create_group('feature_masks')
        # attributes
        f5_masks.attrs['encoding'] = 'raster'
        f5_masks.attrs['encoding-version'] = 'draft_luca'
        # datasets
        f5_masks['cells'] = masks


if True:
    tempdir = 'temp/'
    os.makedirs(tempdir, exist_ok=True)
# with tempfile.TemporaryDirectory() as tempdir:
    print(f'temporary directory in use: {tempdir}')

    f0 = download_file(METADATA_URL, output_dir=tempdir)
    f1 = download_file(OME_AND_MASKS_URL, output_dir=tempdir)

    # f0 = tempdir + 'OMEandSingleCellMasks.zip'
    # f1 = tempdir + 'SingleCell_and_Metadata.zip'
    assert os.path.isfile(f0)
    assert os.path.isfile(f1)
    verify_file(f0, METADATA_HASH)
    verify_file(f1, OME_AND_MASKS_HASH)

    f0 = unzip_file(f0, remove_archive=True)
    basel_metadata = os.path.join(f0, 'Data_publication/BaselTMA/Basel_PatientMetadata.csv')
    zurich_metadata = os.path.join(f0, 'Data_publication/ZurichTMA/Zuri_PatientMetadata.csv')
    assert os.path.isfile(basel_metadata)
    assert os.path.isfile(zurich_metadata)

    from metadata import get_metadata

    df = get_metadata(clean_verbose=True, basel_csv=basel_metadata, zurich_csv=zurich_metadata)

    f1 = unzip_file(f1, remove_archive=True)
    ome_path = os.path.join(f1, 'OMEnMasks/ome.zip')
    masks_path = os.path.join(f1, 'OMEnMasks/Basel_Zuri_masks.zip')

    ome_path = unzip_file(ome_path, remove_archive=True)
    masks_path = unzip_file(masks_path, remove_archive=True)
    ome_path = os.path.join(ome_path, 'ome/')
    masks_path = os.path.join(masks_path, 'Basel_Zuri_masks/')

    for index, row in tqdm(df.iterrows(), total=len(df), desc='creating .h5mu objects'):
        ome_filename = row[0]
        f_ome = os.path.join(ome_path, ome_filename)
        assert os.path.isfile(f_ome)

        masks_filename = ome_filename.replace('_full.tiff', '_full_maks.tiff')
        f_masks = os.path.join(masks_path, masks_filename)
        assert os.path.isfile(f_masks), f_masks

        output_folder = 'generated_muon'
        os.makedirs(output_folder, exist_ok=True)
        outfile = ome_filename.replace('.tiff', '.h5mu')
        outfile = os.path.join(output_folder, outfile)
        create_muon_spatial_object(f_ome, f_masks, outfile)
