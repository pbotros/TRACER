import argparse
import glob
import json
import tempfile
import zipfile

import matplotlib.path
import collections
import re

import matplotlib.pyplot as plt
import numpy as np

from tracer import preprocess_histology, ProbesRegistration, AtlasLoader, vis_registered_probes, ProbeTrace, \
    HistologyTransform


def _preprocess(args):
    histology_folder = args.folder
    fns = []
    for f in os.listdir(histology_folder):
        if not f.endswith('.tif'):
            continue
        prefix = os.path.splitext(f)[0]
        processed = os.path.join(histology_folder, 'processed')
        if os.path.exists(processed) and any([(prefix in f) for f in os.listdir(processed)]):
            continue
        fns.append(f)

    print(len(fns))
    assert len(fns) > 0

    for i, fn in enumerate(fns):
        plt.close('all')
        print(fn)
        h = preprocess_histology(histology_folder=histology_folder, file_name=fn, plane='c')
        if i < len(fns) - 1:
            import time
            time.sleep(1)


def _hex_to_rgb(hex):
    hex = hex.strip('#')
    return tuple((int(hex[i:i + 2], 16)) / 255 for i in (0, 2, 4))
DSRED_COLOR = _hex_to_rgb('#FF8100')
DSRED_COLOR /= np.linalg.norm(DSRED_COLOR, 2)

def _preprocess_filter(args):
    histology_folder = args.folder
    ignored_folder = args.ignore_folder
    if not os.path.exists(ignored_folder):
        os.mkdir(ignored_folder)
    fns_by_im_number = collections.defaultdict(list)

    for f in os.listdir(os.path.join(histology_folder, 'processed')):
        if not f.endswith('.tif'):
            continue
        split = os.path.splitext(f)[0].split('_')
        im_number = int(split[-2])
        slice_number = int(split[-1].replace('s', ''))
        fn = os.path.join(histology_folder, 'processed', f)
        fns_by_im_number[im_number].append((slice_number, fn))

    fns_to_keep = []
    fns_to_ignore = []
    for im_number in sorted(fns_by_im_number.keys()):
        slices = fns_by_im_number[im_number]
        if len(slices) == 1:
            fns_to_keep.append(slices[0][1])
            print(slices[0][1])
            continue
        assert len(slices) == 2

        slice0 = slices[0]
        slice1 = slices[1]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        fig.subplots_adjust(wspace=0.1)

        fn0 = slice0[1]
        im0 = Image.open(fn0)

        fn1 = slice1[1]
        im1 = Image.open(fn1)

        ax[0].set_title(os.path.basename(fn0))
        ax[0].imshow(im0, cmap='gray')

        ax[1].set_title(os.path.basename(fn1))
        ax[1].imshow(im1, cmap='gray')

        ax[0].tick_params(axis='both', labelleft=False, labelbottom=False, left=False, bottom=False)
        ax[1].tick_params(axis='both', labelleft=False, labelbottom=False, left=False, bottom=False)
        fig.tight_layout()

        def _on_click(event, selected):
            if event.inaxes == ax[0]:
                selected[0] = 0
            else:
                selected[0] = 1
            plt.close(fig)

        selected = [None]
        fig.canvas.mpl_connect('button_press_event',
                               lambda event: _on_click(event, selected=selected))
        plt.show()
        if selected[0] == 0:
            fns_to_keep.append(fn0)
            fns_to_ignore.append(fn1)
            print(fn0)
        elif selected[0] == 1:
            fns_to_keep.append(fn1)
            fns_to_ignore.append(fn0)
            print(fn1)
        else:
            print('Neither selected??? Breaking.')
            break

    for fn in fns_to_ignore:
        os.rename(fn, os.path.join(ignored_folder, os.path.basename(fn)))

    new_fns = []
    for fn in fns_to_keep:
        # Ensure it's in the right format:
        basename = os.path.basename(fn)
        prefix, ext = os.path.splitext(basename)
        parts = prefix.split('__')
        if len(parts) < 2:
            raise ValueError(f'Image was not of the expected format? {fn}')
        prefix_start = '__'.join(parts[:-1])
        prefix_end = parts[-1]
        matches = re.findall('^([0-9]+)_s[0-9]*', prefix_end)
        if len(matches) == 0:
            matches = re.findall('s[0-9]+', prefix_end)
            if len(matches) == 0:
                raise ValueError(f'Could not find appropriate filename numbering in image {fn}!')
            new_fns.append(fn)
        elif len(matches) != 1:
            raise ValueError('Multiple matches?')
        else:
            slide_number_str = matches[0]
            new_basename = f'{prefix_start}__s{slide_number_str}{ext}'
            new_fn = os.path.join(os.path.dirname(fn), new_basename)
            print(f'Moving {fn} to {new_fn}')
            os.rename(fn, new_fn)
            new_fns.append(new_fn)

    # And then create .png and DsRed-projected .png files for each file for use in alignment / segmentation
    for f in os.listdir(os.path.join(histology_folder, 'processed')):
        if not f.endswith('.tif'):
            continue

        fn = os.path.join(histology_folder, 'processed', f)
        im = Image.open(fn)

        im.save(f'{os.path.splitext(fn)[0]}.png')

        (
            Image.fromarray((np.array(im) @ DSRED_COLOR))
                .convert('LA')
                .convert('RGB')
                .save(f'{os.path.splitext(fn)[0]}_dsred.png')
        )

def _registration(args):
    atlas = _atlas(args)
    register_probes = ProbesRegistration(atlas, processed_histology_folder=args.folder, plane='c')

def _viz_2d(args):
    atlas = _atlas(args)
    v = vis_registered_probes(atlas, probe_folder=args.folder, plot=not args.no_plot)
    v.vis2d()

def _viz_3d(args):
    atlas = _atlas(args)
    v = vis_registered_probes(atlas, probe_folder=args.folder)
    v.vis3d()

def _atlas(args):
    from dask.distributed import Client
    client = Client('127.0.0.1:8786')
    if args.atlas_version == 'v3':
        fn = '/home/carmena/WHS_SD_rat_atlas_v3'
    elif args.atlas_version == 'v4':
        fn = '/home/carmena/WHS_SD_rat_atlas_v4'
    else:
        raise ValueError(f'Unknown atlas version {args.atlas_version}')
    return AtlasLoader(fn, args.atlas_version, use_dask=True)


import os
from PIL import Image
import h5py

def _curate_probe_trace(args):
    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        from scipy.spatial import Delaunay
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0

    basenames = sorted(os.listdir(args.folder))
    for i, basename in enumerate(basenames):
        if not basename.endswith('dsred_Simple Segmentation.h5'):
            continue

        fn = os.path.join(args.folder, basename)
        print(f'Processing {fn}...')
        color_fn = fn.replace('_dsred_Simple Segmentation.h5', '.png')
        new_fn = fn.replace('_dsred_Simple Segmentation.h5', '_segmentation.png')
        if os.path.exists(new_fn):
            print(f'{new_fn} already exists. Skipping.')
            continue

        f = h5py.File(fn, 'r')
        target_label = 1
        im = f['exported_data'][:]
        im = im.reshape((im.shape[0], im.shape[1]))
        im_copy = np.zeros_like(im, dtype=np.uint8)
        im_copy[im == target_label] = 255

        fig, ax = plt.subplots(1, 2)
        fig.subplots_adjust(wspace=0.1)
        [a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) for a in ax]
        ax[0].imshow(Image.open(color_fn))

        segmented_im_plotted = ax[1].imshow(im_copy, cmap='gray')

        clicked_pts = []
        plotted_pts = []
        plotted_lines = []
        indices_lists_removed = []

        def _onclick(event):
            if event.inaxes != ax[1]:
                return
            if event.button.name == 'LEFT':
                clicked_pts.append((event.xdata, event.ydata))
                plotted_pts.append(ax[1].scatter(event.xdata, event.ydata, marker='o', color='red', s=3))
                if len(clicked_pts) >= 2:
                    plotted_lines.extend(ax[1].plot(
                        [clicked_pts[-2][0], clicked_pts[-1][0]],
                        [clicked_pts[-2][1], clicked_pts[-1][1]], color='red', lw=0.5))
            else:
                path = matplotlib.path.Path(clicked_pts)
                indices = np.argwhere(im_copy.T > 0)

                indices_outside = indices[~path.contains_points(indices)]
                for index in indices_outside:
                    im_copy[index[1], index[0]] = 0
                indices_lists_removed.append(indices_outside)
                segmented_im_plotted.set_array(im_copy)
                clicked_pts.clear()
                [pt.remove() for pt in plotted_pts]
                [line.remove() for line in plotted_lines]
                plotted_pts.clear()
                plotted_lines.clear()
            fig.canvas.draw()

        def _onkey(event):
            if event.key != 'r':
                return

            if len(indices_lists_removed) == 0:
                return
            indices_to_add_back = indices_lists_removed.pop(-1)
            for index in indices_to_add_back:
                im_copy[index[1], index[0]] = 255
            segmented_im_plotted.set_array(im_copy)
            fig.canvas.draw()

        fig.tight_layout()
        fig.canvas.mpl_connect('button_press_event', _onclick)
        fig.canvas.mpl_connect('key_press_event', _onkey)
        plt.show(block=True)

        im_copy = Image.fromarray(im_copy, mode='L').convert('RGB')
        print(f'Saving trimmed and renamed image to {new_fn}.')
        im_copy.save(new_fn)

        if i < len(basenames) - 1:
            import time;
            time.sleep(1)

def _package(args):
    # angles.json
    atlas_maps_files = [
        '*-Segmentation.flat'
    ]

    segmentation_files = [
        '*_segmentation.png'
    ]

    def _extract_slide_id(file):
        bn = os.path.basename(file)
        matches = re.findall('__s([0-9]+)', bn)
        if len(matches) != 1:
            return None
        return matches[0]

    processed_filenames_for_glob = collections.defaultdict(set)
    for zip_dir, file_globs in (
            ('atlas_maps', atlas_maps_files),
            ('segmentation', segmentation_files),
    ):
        for file_glob in file_globs:
            for file in glob.glob(os.path.join(args.folder, file_glob)):
                slide_id = _extract_slide_id(file)
                if slide_id is None:
                    continue
                processed_filenames_for_glob[zip_dir].add(slide_id)

    slide_ids_to_keep = None
    for ids in processed_filenames_for_glob.values():
        if slide_ids_to_keep is None:
            slide_ids_to_keep = ids
        else:
            slide_ids_to_keep &= ids
    assert slide_ids_to_keep
    with open(os.path.join(args.folder, 'angles.json'), 'r') as f:
        angles_json = json.loads(f.read())
        for slice_dict in angles_json['slices']:
            slide_id = _extract_slide_id(slice_dict['filename'])
            if 'anchoring' not in slice_dict or len(slice_dict['anchoring']) == 0:
                slide_ids_to_keep -= {slide_id}

    print(f'Packaging slide IDs:', sorted(slide_ids_to_keep))

    with open(os.path.join(args.folder, 'angles.json'), 'r') as f:
        src = f.read()
        angles_json = json.loads(src)
        angles_json_copy = json.loads(src)
        angles_json_copy['slices'] = []
        for slice_dict in angles_json['slices']:
            if _extract_slide_id(slice_dict['filename']) in slide_ids_to_keep:
                angles_json_copy['slices'].append(slice_dict)

    output_zip_file = os.path.join(args.folder, 'processed.zip')

    with tempfile.TemporaryDirectory(dir=args.folder) as tmpdir:
        angles_json_copy_fn = os.path.join(os.path.basename(tmpdir), 'angles.json')
        with open(os.path.join(args.folder, angles_json_copy_fn), 'w') as f:
            f.write(json.dumps(angles_json_copy))

        with zipfile.ZipFile(output_zip_file, "w") as zf:
            for zip_dir, file_globs in (
                    ('atlas_maps', atlas_maps_files),
                    ('segmentation', segmentation_files),
                    ('anchoring_files', (angles_json_copy_fn,)),
            ):
                has_slide_ids = zip_dir in processed_filenames_for_glob.keys()
                for file_glob in file_globs:
                    for f in sorted(glob.glob(os.path.join(args.folder, file_glob))):
                        if not has_slide_ids or _extract_slide_id(f) in slide_ids_to_keep:
                            print('Including', f)
                            zf.write(
                                filename=os.path.join(args.folder, f),
                                arcname=os.path.join('processed', zip_dir, os.path.basename(f))
                            )
    print(f'Wrote to zip file {output_zip_file}')

def _postprocess(args):
    coordinates_json_file = args.coordinates_json
    with open(coordinates_json_file, 'r') as f:
        coordinates_json = json.loads(f.read())
    coordinates = []
    for dd in coordinates_json:
        coordinates.append(np.array(dd['triplets'], dtype=np.float64).reshape((-1, 3)))
    coordinates = np.vstack(coordinates)

    print(f'Wrote {coordinates.shape[0]} coordinates.')
    with open(args.output_json, 'w') as f:
        f.write(ProbeTrace(
            histology_transform=HistologyTransform(
                histology_filename='combined.tif',
                transform_atlas_plane='c',
                transform_points_histology=[],
                transform_points_atlas=[],
                transform_atlas_slice_index=int(np.median(coordinates[:, 1].ravel())),
                transform_atlas_angle=0,
            ),
            probe_name='probe0',
            color='purple',
            probe_atlas_coords=[tuple(k) for k in coordinates.tolist()],
        ).to_json(indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.set_defaults(command=None)

    subparsers = parser.add_subparsers()
    parser_preprocess = subparsers.add_parser('preprocess')
    parser_preprocess.add_argument("folder", type=str, help='Histology folder containing *tif/*jpg')
    parser_preprocess.set_defaults(command=_preprocess)

    parser_preprocess_filter = subparsers.add_parser('preprocess-filter')
    parser_preprocess_filter.add_argument("folder", type=str, help='Histology folder containing *tif/*jpg')
    parser_preprocess_filter.add_argument("--ignore-folder", type=str, required=True,
                                          help='Folder in which to move ignored files.')
    parser_preprocess_filter.set_defaults(command=_preprocess_filter)

    parser_registration = subparsers.add_parser('registration')
    parser_registration.add_argument("folder", type=str, help='Histology folder containing *tif/*jpg')
    parser_registration.add_argument("--atlas-version", type=str, help='Atlas version', default='v3')
    parser_registration.set_defaults(command=_registration)

    parser_curation = subparsers.add_parser('curate')
    parser_curation.add_argument("folder", type=str, help='Histology folder containing *h5')
    parser_curation.set_defaults(command=_curate_probe_trace)

    parser_package = subparsers.add_parser('package')
    parser_package.add_argument("folder", type=str, help='Histology folder containing the _segmentation.png, angles.json/angles.xml,')
    parser_package.set_defaults(command=_package)

    parser_postprocess = subparsers.add_parser('postprocess')
    parser_postprocess.add_argument("coordinates_json", type=str, help='Coordinates combined json file')
    parser_postprocess.add_argument("output_json", type=str, help='probe file json to write')
    parser_postprocess.set_defaults(command=_postprocess)

    parser_viz = subparsers.add_parser('viz')
    parser_viz.add_argument("folder", type=str, help='Histology folder containing probes folder')
    parser_viz.add_argument("--atlas-version", type=str, help='Atlas version', default='v3')
    parser_viz.add_argument("--no-plot", action='store_true', default=False, help='Whether a plot should be formed')
    parser_viz.set_defaults(command=_viz_2d)

    parser_viz3d = subparsers.add_parser('viz3d')
    parser_viz3d.add_argument("folder", type=str, help='Histology folder containing probes folder')
    parser_viz3d.add_argument("--atlas-version", type=str, help='Atlas version', default='v3')
    parser_viz3d.set_defaults(command=_viz_3d)

    args = parser.parse_args()

    if args.command is None:
        raise ValueError('Need to specify a command!')
    args.command(args)
