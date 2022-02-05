import numpy as np
import os
import nibabel as nib
import cv2


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

    def __init__(self, atlas_folder, atlas_version='v4'):
        if not os.path.exists(atlas_folder):
            raise Exception('Folder path %s does not exist. Please give the correct path.' % atlas_folder)

        atlas_data_masked_path = os.path.join(atlas_folder, 'atlas_data_masked.npz')
        if os.path.exists(atlas_data_masked_path):
            with np.load(atlas_data_masked_path) as temp_data:
                self.atlas_data = temp_data['atlas_data']
                self.pixdim = temp_data['pixdim']
                self.mask_data = temp_data['mask_data']
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
            self.pixdim = atlas_header.get('pixdim')[1]
            self.atlas_data = atlas.get_fdata()

            # Load Mask
            print('Loading mask...')
            mask = nib.load(mask_path)
            self.mask_data = mask.get_fdata()[:, :, :, 0]

            # Remove skull and tissues from atlas_data #
            CC = np.where(self.mask_data == 0)
            self.atlas_data[CC] = 0
            atlas = {'atlas_data': self.atlas_data, 'pixdim': self.pixdim, 'mask_data': self.mask_data}

            print('Dumping to disk...')
            np.savez_compressed(
                atlas_data_masked_path,
                atlas_data=self.atlas_data,
                pixdim=self.pixdim,
                mask_data=self.mask_data)
            print('Done precomputing mask file.')


        # check other path
        segmentation_path = os.path.join(atlas_folder, 'WHS_SD_rat_atlas_%s.nii.gz' % atlas_version)
        label_path = os.path.join(atlas_folder, 'WHS_SD_rat_atlas_%s.label' % atlas_version)

        if not os.path.exists(segmentation_path):
            raise Exception('Segmentation file %s does not exist. Please download the segmentation data.' % segmentation_path)

        if not os.path.exists(label_path):
            raise Exception('Label file %s does not exist. Please download the label data.' % label_path)

        # Load Segmentation
        segmentation = nib.load(segmentation_path)
        self.segmentation_data = segmentation.get_fdata()

        # Load Labels
        with open(label_path, "r") as labels_item:
            self.labels_index, self.labels_name, self.labels_color, self.labels_initial = readlabel(labels_item)


        cv_plot_path = os.path.join(atlas_folder, 'cv_plot.npz')
        edges_path = os.path.join(atlas_folder, 'Edges.npz')
        cv_plot_disp_path = os.path.join(atlas_folder, 'cv_plot_display.npz')


        if os.path.exists(cv_plot_path):
            self.cv_plot = np.load(cv_plot_path)['cv_plot'] / 255
        else:
            print('Precomputing cv_plot...')
            # Create an inverted map so we can index below quickly
            labels_index_flat = np.zeros(np.max(self.labels_index.max()) + 1, dtype=np.int64)
            labels_index_flat[self.labels_index.ravel()] = np.arange(self.labels_color.shape[0])

            # Atlas in RGB colors according with the label file #
            self.cv_plot = self.labels_color[labels_index_flat[self.segmentation_data.astype(np.int64).ravel()]].reshape(tuple(list(self.segmentation_data.shape) + [3]))
            np.savez_compressed(cv_plot_path, cv_plot=self.cv_plot)
            self.cv_plot = self.cv_plot / 255

        if os.path.exists(edges_path):
            self.Edges = np.load(edges_path)['Edges']
        else:
            print('Precomputing edges...')
            # Get the edges of the colors defined in the label #
            self.Edges = np.empty((512, 1024, 512))
            for sl in range(0, 1024):
                self.Edges[:, sl, :] = cv2.Canny(np.uint8((self.cv_plot[:, sl, :] * 255).transpose((1, 0, 2))), 100, 200)
            np.savez_compressed(edges_path, Edges=self.Edges)


        if os.path.exists(cv_plot_disp_path):
            self.cv_plot_display = np.load(cv_plot_disp_path)['cv_plot_display']
        else:
            print('Precomputing cv_plot_disp...')

            # Create an inverted map so we can index below quickly
            labels_index_flat = np.zeros(np.max(self.labels_index.max()) + 1, dtype=np.int64)
            labels_index_flat[self.labels_index.ravel()] = np.arange(self.labels_color.shape[0])

            labels_color_augmented = self.labels_color.copy()
            labels_color_augmented[0, :] = [128, 128, 128]

            # here I create the array to plot the brain regions in the RGB
            # of the label file
            self.cv_plot_display = labels_color_augmented[labels_index_flat[self.segmentation_data.astype(np.int64).ravel()]].reshape(tuple(list(self.segmentation_data.shape) + [3]))
            np.savez_compressed(cv_plot_disp_path, cv_plot_display=self.cv_plot_display)

