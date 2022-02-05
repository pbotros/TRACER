import os
from typing import Optional, Dict

import cv2
import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def readlabel(file):
    """
    Purpose
    -------------
    Read the atlas label file.

    Inputs
    -------------
    file :

    Outputs
    -------------
    A list contains ...

    """
    output_index = []
    output_names = []
    output_colors = []
    output_initials = []
    labels = file.readlines()
    pure_labels = [x for x in labels if "#" not in x]

    for line in pure_labels:
        line_labels = line.split()
        accessed_mapping = map(line_labels.__getitem__, [0])
        L = list(accessed_mapping)
        indice = [int(i) for i in L]
        accessed_mapping_rgb = map(line_labels.__getitem__, [1, 2, 3])
        L_rgb = list(accessed_mapping_rgb)
        colorsRGB = [int(i) for i in L_rgb]
        output_colors.append(colorsRGB)
        output_index.append(indice)
        output_names.append(' '.join(line_labels[7:]))
    for i in range(len(output_names)):
        output_names[i] = output_names[i][1:-1]
        output_initials_t = []
        for s in output_names[i].split():
            output_initials_t.append(s[0])
        output_initials.append(' '.join(output_initials_t))
    output_index = np.array(output_index)  # Use numpy array for  in the label file
    output_colors = np.array(output_colors)  # Use numpy array for  in the label file
    return [output_index, output_names, output_colors, output_initials]


def _precompute_data(atlas_folder, atlas_version):
    if not os.path.exists(atlas_folder):
        raise Exception('Folder path %s does not exist. Please give the correct path.' % atlas_folder)

    atlas_data_masked_path = os.path.join(atlas_folder, 'atlas_data_masked.npz')
    if os.path.exists(atlas_data_masked_path):
        print('Atlas data already precomputed.')
    else:
        atlas_path = os.path.join(atlas_folder, 'WHS_SD_rat_T2star_v1.01.nii.gz')
        mask_path = os.path.join(atlas_folder, 'WHS_SD_rat_brainmask_v1.01.nii.gz')

        if not os.path.exists(atlas_path):
            raise Exception('Atlas file %s does not exist. Please download the atlas data.' % atlas_path)

        if not os.path.exists(mask_path):
            raise Exception('Mask file %s does not exist. Please download the mask data.' % mask_path)

        print(f'Precomputed mask {atlas_data_masked_path} not present; computing and caching on disk. This may take awhile...')
        # Load Atlas
        print('Loading atlas...')
        atlas = nib.load(atlas_path)
        atlas_header = atlas.header
        pixdim = atlas_header.get('pixdim')[1]
        atlas_data = atlas.get_fdata()

        # Load Mask
        print('Loading mask...')
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata()[:, :, :, 0]

        # Remove skull and tissues from atlas_data #
        CC = np.where(mask_data == 0)
        atlas_data[CC] = 0

        print('Dumping to disk...')
        np.savez_compressed(
            atlas_data_masked_path,
            atlas_data=atlas_data,
            pixdim=pixdim,
            mask_data=mask_data)
        print('Done precomputing mask file.')

    # check other path
    segmentation_path = os.path.join(atlas_folder, 'WHS_SD_rat_atlas_%s.nii.gz' % atlas_version)
    label_path = os.path.join(atlas_folder, 'WHS_SD_rat_atlas_%s.label' % atlas_version)

    if not os.path.exists(segmentation_path):
        raise Exception('Segmentation file %s does not exist. Please download the segmentation data.' % segmentation_path)

    if not os.path.exists(label_path):
        raise Exception('Label file %s does not exist. Please download the label data.' % label_path)

    # Load Segmentation
    segmentation_data = None

    # Load Labels
    with open(label_path, "r") as labels_item:
        labels_index, labels_name, labels_color, labels_initial = readlabel(labels_item)

    cv_plot_path = os.path.join(atlas_folder, 'cv_plot.npz')
    edges_path = os.path.join(atlas_folder, 'Edges.npz')
    cv_plot_disp_path = os.path.join(atlas_folder, 'cv_plot_display.npz')

    cv_plot = None
    if os.path.exists(cv_plot_path):
        pass
    else:
        print('Precomputing cv_plot...')
        if segmentation_data is None:
            segmentation = nib.load(segmentation_path)
            segmentation_data = segmentation.get_fdata()
        # Create an inverted map so we can index below quickly
        labels_index_flat = np.zeros(np.max(labels_index.max()) + 1, dtype=np.int64)
        labels_index_flat[labels_index.ravel()] = np.arange(labels_color.shape[0])

        # Atlas in RGB colors according with the label file #
        cv_plot = labels_color[labels_index_flat[segmentation_data.astype(np.int64).ravel()]].reshape(tuple(list(segmentation_data.shape) + [3]))
        cv_plot = cv_plot.astype(np.float64)
        cv_plot /= 255
        np.savez_compressed(cv_plot_path, cv_plot=cv_plot)

    if os.path.exists(edges_path):
        pass
    else:
        print('Precomputing edges...')
        if cv_plot is None:
            cv_plot = np.load(cv_plot_path)['cv_plot']

        # Get the edges of the colors defined in the label #
        Edges = np.empty((512, 1024, 512))
        for sl in range(0, 1024):
            Edges[:, sl, :] = cv2.Canny(np.uint8((cv_plot[:, sl, :] * 255).transpose((1, 0, 2))), 100, 200)
        np.savez_compressed(edges_path, Edges=Edges)

    if os.path.exists(cv_plot_disp_path):
        pass
    else:
        print('Precomputing cv_plot_disp...')

        # Create an inverted map so we can index below quickly
        labels_index_flat = np.zeros(np.max(labels_index.max()) + 1, dtype=np.int64)
        labels_index_flat[labels_index.ravel()] = np.arange(labels_color.shape[0])

        labels_color_augmented = labels_color.copy()
        labels_color_augmented[0, :] = [128, 128, 128]

        # here I create the array to plot the brain regions in the RGB
        # of the label file
        if segmentation_data is None:
            segmentation = nib.load(segmentation_path)
            segmentation_data = segmentation.get_fdata()
        cv_plot_display = labels_color_augmented[labels_index_flat[segmentation_data.astype(np.int64).ravel()]].reshape(tuple(list(segmentation_data.shape) + [3]))
        np.savez_compressed(cv_plot_disp_path, cv_plot_display=cv_plot_display)

def _load_pixdim(atlas_folder, atlas_version):
    # Assumes that _precompute_atlas_data() has already been done.
    with np.load(os.path.join(atlas_folder, 'atlas_data_masked.npz')) as temp_data:
        return float(temp_data['pixdim'])

def _load_atlas_data(atlas_folder, atlas_version):
    # Assumes that _precompute_atlas_data() has already been done.
    with np.load(os.path.join(atlas_folder, 'atlas_data_masked.npz')) as temp_data:
        return temp_data['atlas_data']

def _load_mask_data(atlas_folder, atlas_version):
    # Assumes that _precompute_atlas_data() has already been done.
    with np.load(os.path.join(atlas_folder, 'atlas_data_masked.npz')) as temp_data:
        return temp_data['mask_data']

def _load_segmentation(atlas_folder, atlas_version):
    # Assumes that _precompute_atlas_data() has already been done.
    segmentation_path = os.path.join(atlas_folder, 'WHS_SD_rat_atlas_%s.nii.gz' % atlas_version)
    segmentation = nib.load(segmentation_path)
    return segmentation.get_fdata()

def _load_edges(atlas_folder, atlas_version):
    # Assumes that _precompute_atlas_data() has already been done.
    with np.load(os.path.join(atlas_folder, 'Edges.npz')) as temp_data:
        return temp_data['Edges']

def _load_cv_plot(atlas_folder, atlas_version):
    # Assumes that _precompute_atlas_data() has already been done.
    with np.load(os.path.join(atlas_folder, 'cv_plot.npz')) as temp_data:
        return temp_data['cv_plot']

def _load_cv_plot_display(atlas_folder, atlas_version):
    # Assumes that _precompute_atlas_data() has already been done.
    with np.load(os.path.join(atlas_folder, 'cv_plot_display.npz')) as temp_data:
        return temp_data['cv_plot_display']


def _load_labels_index_name(atlas_folder, atlas_version):
    label_path = os.path.join(atlas_folder, 'WHS_SD_rat_atlas_%s.label' % atlas_version)
    # Load Labels
    with open(label_path, "r") as labels_item:
        labels_index, labels_name, labels_color, labels_initial = readlabel(labels_item)
        return labels_index, labels_name, labels_color, labels_initial


def _compute_angled_slice(data, x, y, z, angle):
    # Only y indexing supported right now with angle because laziness
    assert y is not None
    assert x is None
    assert z is None

    shape = data.shape
    interpolator = RegularGridInterpolator(
        (np.arange(shape[0]), (np.arange(shape[1])), (np.arange(shape[2]))), data)

    xg, zg = np.meshgrid(np.arange(shape[0]), np.arange(shape[2]), indexing='ij')
    y_span_half = int(round(np.tan(angle / 180 * np.pi) * shape[0] / 2))
    y_span = (y - y_span_half, y + y_span_half)
    coords = np.vstack((
        xg.ravel(),
        np.repeat(np.linspace(y_span[0], y_span[1], shape[0]), shape[2]),
        zg.ravel())).T
    interped = interpolator(coords)
    return interped.reshape((shape[0], shape[2]))


def _load_edges_coords(edges):
    edges = edges.T
    edges[:, ::2, :] = edges[:, ::2, :] * 0
    edges[:, ::5, :] = edges[:, ::5, :] * 0
    edges[::2, :, :] = edges[::2, :, :] * 0
    edges[:, :, ::2] = edges[:, :, ::2] * 0

    return np.array(np.where(edges == 255))


class AtlasLoader(object):
    """
    Purpose
    -------------
    Load the Waxholm Rat Brain Atlas.

    Inputs
    -------------
    atlas_folder  : str, the path of the folder contains all the atlas related data files.
    atlas_version : str, specify the version of atlas in use, default is version 3 ('v3').

    Outputs
    -------------
    Atlas object.

    """

    def __init__(self, atlas_folder, atlas_version='v4', use_dask=False):
        self._use_dask = use_dask
        if use_dask:
            import dask.array as da
            import dask
            dask.delayed(_precompute_data, pure=True)(atlas_folder, atlas_version).compute()

            self.atlas_data = da.from_delayed(
                dask.delayed(_load_atlas_data, pure=True)(atlas_folder, atlas_version),
                shape=(512, 1024, 512), dtype=np.float64).persist()
            self.pixdim = dask.delayed(_load_pixdim, pure=True)(atlas_folder, atlas_version).compute()
            self.mask_data = da.from_delayed(
                dask.delayed(_load_mask_data, pure=True)(atlas_folder, atlas_version),
                shape=(512, 1024, 512), dtype=np.float64).persist()
            self.segmentation_data = da.from_delayed(
                dask.delayed(_load_segmentation, pure=True)(atlas_folder, atlas_version),
                shape=(512, 1024, 512), dtype=np.float64).persist()
            self.cv_plot = da.from_delayed(
                dask.delayed(_load_cv_plot, pure=True)(atlas_folder, atlas_version),
                shape=(512, 1024, 512, 3), dtype=np.float64).persist()
            self.cv_plot_display = da.from_delayed(
                dask.delayed(_load_cv_plot_display, pure=True)(atlas_folder, atlas_version),
                shape=(512, 1024, 512, 3), dtype=np.int64).persist()
            self.Edges = da.from_delayed(
                dask.delayed(_load_edges, pure=True)(atlas_folder, atlas_version),
                shape=(512, 1024, 512), dtype=np.float64).persist()

            self._labels_index, self._labels_name, self._labels_color, self._labels_initial = dask.delayed(
                _load_labels_index_name, pure=True)(atlas_folder, atlas_version).compute()

            # And then trigger a dummy slice of atlas data to block for it to load
            from distributed import Client
            Client.current().wait_for_workers(1)
        else:
            _precompute_data(atlas_folder, atlas_version)
            self.atlas_data = _load_atlas_data(atlas_folder, atlas_version)
            self.pixdim = _load_pixdim(atlas_folder, atlas_version)
            self.mask_data = _load_mask_data(atlas_folder, atlas_version)
            self.segmentation_data = _load_segmentation(atlas_folder, atlas_version)
            self.cv_plot = _load_cv_plot(atlas_folder, atlas_version)
            self.cv_plot_display = _load_cv_plot_display(atlas_folder, atlas_version)
            self.Edges = _load_edges(atlas_folder, atlas_version)
            self._labels_index, self._labels_name, self._labels_color, self._labels_initial = \
                _load_labels_index_name(atlas_folder, atlas_version)

        # ??
        # self._bregma = (266, 653, 440)
        self._bregma = (266, 623, 440)

    @property
    def bregma_y_index(self):
        return self._bregma[1]

    @property
    def bregma_x_index(self):
        return self._bregma[0]

    @property
    def bregma_z_index(self):
        return self._bregma[2]

    @property
    def shape(self):
        return 512, 1024, 512,

    @property
    def labels_color(self):
        return self._labels_color

    @property
    def labels_name(self):
        return self._labels_name

    def segmentation_region(self, x, y, z) -> Optional[Dict]:
        x = int(x)
        y = int(y)
        z = int(z)
        for i, coord in enumerate((x, y, z)):
            if coord < 0 or coord >= self.shape[i]:
                return None
        segmentation_slice = self.segmentation_data[x, y, z]
        if self._use_dask:
            segmentation_slice = segmentation_slice.compute()

        indices = np.argwhere(np.all(self._labels_index == segmentation_slice, axis=1)).ravel()
        if len(indices) == 0:
            return None
        else:
            index = indices[0]
            return {
                'name': self._labels_name[index],
                'initial': self._labels_initial[index],
                'color': self._labels_color[index],
            }

    def edge_plane(self, x=None, y=None, z=None):
        return self._data_plane(data=self.Edges, x=x, y=y, z=z, angle=None)

    def plane(self, x=None, y=None, z=None, angle=None):
        return self._data_plane(data=self.atlas_data, x=x, y=y, z=z, angle=angle)

    def cv_plot_plane(self, x=None, y=None, z=None):
        return self._data_plane(data=self.cv_plot, x=x, y=y, z=z, angle=None)

    def cv_plot_display_plane(self, x=None, y=None, z=None):
        return self._data_plane(data=self.cv_plot_display, x=x, y=y, z=z, angle=None)

    def edges_coords(self):
        if self._use_dask:
            import dask
            return dask.delayed(_load_edges_coords, pure=True)(self.Edges).compute()
        else:
            return _load_edges_coords(self.Edges)

    def _data_plane(self, data, x, y, z, angle):
        assert (x is not None) or (y is not None) or (z is not None)
        if angle is None:
            angle = 0
        if abs(angle) < 1e-6:
            ret = data[
                x if x is not None else slice(x),
                y if y is not None else slice(y),
                z if z is not None else slice(z)]
            if self._use_dask:
                return ret.compute()
            else:
                return ret
        else:
            # Only y indexing supported right now with angle because laziness
            if self._use_dask:
                import dask
                return dask.delayed(_compute_angled_slice, pure=True)(data, x, y, z, angle).compute()
            else:
                return _compute_angled_slice(data, x, y, z, angle)
