#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:18:44 2021

@author: admin
"""


from __future__ import print_function

import math
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from PIL import Image
from six.moves import input
from skimage import transform
from skspatial.objects import Line

# create objects to svae transformations and probes
from .ObjSave import ProbeTrace, HistologyTransform
from .index_tracker import IndexTracker, IndexTracker_g, IndexTracker_p


# Import libraries


class ProbesRegistration(object):
    """
    Purpose
    -------------
    For register probes.

    Inputs
    -------------
    atlas :
    processed_histology_folder :
    show_hist :
    probe_name :
    

    Outputs
    -------------
    A list contains ...

    """

    _FIGSIZE = (8, 8)
    _PROBE_COLORS = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']

    def __init__(self, atlas, processed_histology_folder, show_hist=True, probe_name='probe', plane=None):
        
        self.atlas = atlas
        self.processed_histology_folder = processed_histology_folder
        self.show_hist = show_hist
        self.probe_name = probe_name

        # Lists for the points plotted in atlas and histology
        self.redp_atlas = []
        self.redp_hist = []
        self.rednum_atlas = []
        self.rednum_hist = []

        # List of probe points
        self.p_probe_trans = []
        self.p_probe_grid = []

        # Initialize probe counter and selecter
        self.probe_counter = 0
        self.probe_selecter = 0
        self.flag = 0

        # For simplicity: are there even valid flows for which either of these are false?
        assert processed_histology_folder is not None
        assert show_hist is not None

        # probes have different colors; match with list
        if not os.path.exists(self.processed_histology_folder):
            raise ValueError('Please give the correct folder.')

        # The Transformed images will be saved in a subfolder of process histology called transformations
        self.path_transformed = os.path.join(self.processed_histology_folder, 'transformations')
        if not os.path.exists(self.path_transformed):
            os.mkdir(self.path_transformed)

        self.path_probes = os.path.join(self.processed_histology_folder, 'probes')
        if not os.path.exists(self.path_probes):
            os.mkdir(self.path_probes)

        # Insert the plane of interest
        self.plane = plane
        if self.plane is None:
            self.plane = str(input('Select the plane coronal (c), sagittal (s), or horizontal (h): ')).lower()
            # Check if the input is correct
            while self.plane != 'c' and self.plane != 's' and self.plane != 'h':
                print('Error: Wrong plane name \n')
                self.plane = str(input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')).lower()

        self._filenames_to_process = []

        for fname in sorted(os.listdir(self.processed_histology_folder)):
            image_path = os.path.join(self.processed_histology_folder, fname)
            if any([image_path.endswith(suffix) for suffix in ('.jpg', '.jpeg', '.tiff', '.tif')]):
                self._filenames_to_process.append(fname)
        if len(self._filenames_to_process) == 0:
            raise ValueError(f'No files found in {self.processed_histology_folder} with proper suffixes.')

        # Start with the first file
        self._filename_index = 0
        self._initialize_new_file()

        # Display the ATLAS resolution
        self.dpi_atl = float(25.4 / self.atlas.pixdim)
        # Bregma coordinates
        self.textstr = f'Bregma (mm): c = {self.atlas.bregma_y_index * self.atlas.pixdim:.3f}, ' \
                       f'h = {440 * self.atlas.pixdim:.3f}, ' \
                       f's = {246 * self.atlas.pixdim:.3f} \n' \
                       f'Bregma (voxels): c = {self.atlas.bregma_y_index}, h = 440, s = 246'
        # these are matplotlib.patch.Patch properties
        self.props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # Figure

        print('\nControls: \n')
        print('--------------------------- \n')
        print('u: load saved transform and atlas location \n')
        print('t: activate mode where clicks are logged for transform \n')
        print('d: delete most recent transform point \n')
        print('h: transform histology to adapt to the atlas \n')
        print("b: simple overlay to scroll through brain regions \n")
        print('p: overlay to the atlas \n')
        print('x: save transform and current atlas location \n')
        print('v: activate color atlas mode \n\n')
        print('r: activate mode where clicks are logged for probe \n')
        print('c: delete most recent probe point \n')
        print('n: add a new probe \n')
        print('w: enable probe viewer mode for current probe  \n')
        print('e: save current probe \n')
        print('--------------------------- \n')

        for key in plt.rcParams.keys():
            if key.startswith('keymap'):
                plt.rcParams[key] = ''
        plt.ioff()

        self.fig, (self.ax, self.ax_hist) = plt.subplots(1, 2, figsize=(self._FIGSIZE[0] * 2, self._FIGSIZE[1]))
        self.fig_hist = self.fig
        # scroll cursor
        self.tracker = IndexTracker(self.ax, self.atlas.pixdim, self.plane, atlas)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        # place a text box with bregma coordinates in bottom left in axes coords
        self.ax.text(0.03, 0.03, self.textstr, transform=self.ax.transAxes, fontsize=6, verticalalignment='bottom', bbox=self.props)
        
        if self.plane == 'c':
            # dimensions
            self.d1 = 512
            self.d2 = 512
        elif self.plane == 's':
            # dimensions
            self.d1 = 1024
            self.d2 = 512
        elif self.plane == 'h':
            # dimensions
            self.d1 = 512
            self.d2 = 1024
        self.ax.format_coord = self.tracker.format_coord
        # Fix size and location of the figure window
        self._set_geometry_current_fig(800, 300, self.d1, self.d2)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.plot_hist()
        self.fig_hist.canvas.mpl_connect('key_press_event', self.on_key)
        # Set up the figure
        # plt.ioff()
        self.fig_trans, self.ax_trans = plt.subplots(1, 1, figsize=self._FIGSIZE)
        self._set_geometry_current_fig(200, 350, self.d2, self.d1)
        self.fig_trans.canvas.mpl_connect('key_press_event', self.on_key)

        self.fig_grid, self.ax_grid = plt.subplots(1, 1, figsize=self._FIGSIZE)
        self.fig_g, self.ax_g = plt.subplots(1, 1, figsize=self._FIGSIZE)
        self.fig_color, self.ax_color = plt.subplots(1, 1, figsize=self._FIGSIZE)
        self.fig_probe, self.ax_probe = plt.subplots(1, 1, figsize=self._FIGSIZE)
        self.fig_grid.set_visible(False)
        self.fig_g.set_visible(False)
        self.fig_color.set_visible(False)
        self.fig_probe.set_visible(False)
        self.fig.tight_layout()
        plt.show(block=True)

    def _initialize_new_file(self):
        if self._filename_index >= len(self._filenames_to_process):
            print('\nNo more histology slice to register')
            plt.close('all')
            return
        fname = self._filenames_to_process[self._filename_index]
        print(f'Processing file {fname}.')

        self._cur_histology_transform = HistologyTransform(
            transform_atlas_plane=self.plane,
            histology_filename=fname,
        )
        self._probe_traces = [ProbeTrace(
            probe_name=self.probe_name + str(i),
            color=probe_color,
            histology_transform=self._cur_histology_transform,
        ) for i, probe_color in enumerate(self._PROBE_COLORS)]
        self._probe_trace_index = 0
        self._cur_probe_trace = self._probe_traces[self._probe_trace_index]

        image_path = os.path.join(self.processed_histology_folder, fname)
        self.img_hist_temp = Image.open(image_path).copy()
        self.probe_selecter = 0

    def _set_geometry_current_fig(self, posx, posy, width, height):
        if 'QT' not in plt.get_backend():
            return
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(posx, posy, width, height)

    def plot_hist(self):
        # get the pixel dimension
        self.dpi_hist = float(self.img_hist_temp.info['dpi'][1])
        self.pixdim_hist = 25.4 / self.dpi_hist  # 1 inch = 25,4 mm
        # Show the HISTOLOGY
        # Set up figure
        # self.fig_hist, self.ax_hist = plt.subplots(1, 1, figsize=self._FIGSIZE)
        # self.fig_hist, self.ax_hist = plt.subplots(1, 1, figsize=(
        # float(self.img_hist[self.jj].shape[1]) / self.dpi_hist, float(self.img_hist[self.jj].shape[0]) / self.dpi_hist))
        self.ax_hist.set_title("Histology viewer")
        # Show the histology image
        self.ax_hist.imshow(
            self.img_hist_temp,
            extent=[0, self.img_hist_temp.width * self.pixdim_hist,
                    self.img_hist_temp.height * self.pixdim_hist, 0])
        # Remove axes tick
        self.ax_hist.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off
        # Remove cursor position
        self.ax_hist.format_coord = lambda x, y: ''
        self.fig_hist.canvas.draw()
        # Fix size and location of the figure window
        self._set_geometry_current_fig(150, 300, self.d1, self.d2)

    def onclick_atlas_hist(self, event):
        if event.xdata is None or event.ydata is None:
            # Outside bounds
            return
        if event.inaxes == self.ax_hist:
            return self.onclick_hist(event)
        else:
            return self.onclick_atlas(event)

    def onclick_atlas(self, event):
        # Mouse click function to store coordinates. Leave a red dot when a point is clicked, Atlas
        ix, iy = event.xdata / self.atlas.pixdim, event.ydata / self.atlas.pixdim
        self.clicka += 1
        # assign global variable to access outside of function
        self._cur_histology_transform.transform_points_atlas.append((ix, iy))
        self.redp_atlas.extend(self.ax.plot(event.xdata, event.ydata, 'ro', markersize=2))
        self.rednum_atlas.append(self.ax.text(event.xdata, event.ydata, self.clicka, fontsize=8, color='red'))
        self.fig.canvas.draw()
        self.active = 'atlas'

    def onclick_hist(self, event):
        # Mouse click function to store coordinates. Leave a red dot when a point is clicked, HISTOLOGY
        xh, yh = event.xdata / self.pixdim_hist, event.ydata / self.pixdim_hist
        self.clickh += 1
        # assign global variable to access outside of function
        self._cur_histology_transform.transform_points_histology.append((xh, yh))
        self.redp_hist.extend(self.ax_hist.plot(event.xdata, event.ydata, 'ro', markersize=2))
        self.rednum_hist.append(self.ax_hist.text(event.xdata, event.ydata, self.clickh, fontsize=8, color='red'))
        self.fig_hist.canvas.draw()
        self.active = 'hist'

    
    def show_annotation(self, sel):
        # Show the names of the regions
        xi, yi = sel.target / self.atlas.pixdim
        if self.plane == 'c':
            Text = self.atlas.segmentation_region(int(xi), self.tracker.ind, int(yi))
        elif self.plane == 's':
            Text = self.atlas.segmentation_region(self.tracker.ind, int(xi), int(yi))
        elif self.plane == 'h':
            Text = self.atlas.segmentation_region(int(xi), int(yi), self.tracker.ind)
        else:
            raise ValueError(f'Unknown plane {self.plane}')

        if Text is None:
            Text = ' '
        sel.annotation.set_text(Text)

    
    def onclick_probe(self, event):
        # plot  point and register all the clicked points

        self.p_probe_grid.extend(
            self.ax_grid.plot(event.xdata, event.ydata, color=self._cur_probe_trace.color, marker='o', markersize=1))
        self.p_probe_trans.extend(
            self.ax_trans.plot(event.xdata, event.ydata, color=self._cur_probe_trace.color, marker='o', markersize=1))

        self._cur_probe_trace.probe_atlas_coords.append(
            self.tracker.transform_screen_coord(event.xdata, event.ydata))

        self.fig_grid.canvas.draw()
        self.fig_trans.canvas.draw()

    def on_key2(self, event):
        if event.key == 'n':
            # add a new probe, the function in defined in onclick_probe
            if self._probe_trace_index + 1 < len(self._probe_traces):
                self._probe_trace_index += 1
                self._cur_probe_trace = self._probe_traces[self._probe_trace_index]
                print('probe %d added (%s)' % (self._probe_trace_index, self._cur_probe_trace.color))
            else:
                print('Cannot add more probes')

        elif event.key == 'c':
            print('Delete clicked probe point')

            # Deleting a probe point: either delete from the current probe or go back to the previous probe if there
            # aren't any in the current
            cur_probe_points = self._cur_probe_trace.probe_atlas_coords
            if len(cur_probe_points) == 0 and self._probe_trace_index > 0:
                self._probe_trace_index -= 1
                cur_probe_points = self._cur_probe_trace.probe_atlas_coords

            if len(cur_probe_points) > 0:
                cur_probe_points.pop(-1)
                self.p_probe_trans[-1].remove()  # remove the point from the plot
                self.fig_trans.canvas.draw()
                self.p_probe_trans.pop(-1)
                self.p_probe_grid[-1].remove()  # remove the point from the plot
                self.fig_grid.canvas.draw()
                self.p_probe_grid.pop(-1)

    def re_load_images(self, histology_basename):
        # if the histology and atlas have been already overlayed in a previous study it is possible to load it and keep working from that stage
        # and start recording the probes
        print('\nLoad image and slice')
        fmn = os.path.join(self.path_transformed, histology_basename + '.json')
        with open(fmn, 'r') as f:
            self._cur_histology_transform = HistologyTransform.from_json(f.read())

        self.tracker.ind = self._cur_histology_transform.transform_atlas_slice_index
        self.tracker.angle = self._cur_histology_transform.transform_atlas_angle
        # open the wreaped figure
        self.ax_trans.imshow(self._cur_histology_transform.img_warped, origin="lower",
                             extent=[0, self.d1 * self.atlas.pixdim, 0, self.d2 * self.atlas.pixdim])
        self.ax_trans.set_title("Histology adapted to atlas")
        # open the overlayed figure
        # self.fig_grid, self.ax_grid = plt.subplots(1, 1)
        self.ax_grid.clear()
        self.fig_grid.set_visible(True)
        self.overlay = self.ax_grid.imshow(self._cur_histology_transform.img_warped_overlay, origin="lower",
                                           extent=[0, self.d1 * self.atlas.pixdim, 0, self.d2 * self.atlas.pixdim])
        self.ax_grid.text(0.15, 0.05, self.textstr, transform=self.ax.transAxes, fontsize=6, verticalalignment='bottom',
                          bbox=self.props)
        self.ax_grid.format_coord = self.tracker.format_coord
        self.ax_grid.set_title("Histology and atlas overlayed")
        self.cursor = mplcursors.cursor(self.overlay, hover=True)
        self.cursor.connect('add', self.show_annotation)
        self._set_geometry_current_fig(200, 350, self.d2, self.d1)
        self.flag = 1
        print('Image loaded')

    def onscroll(self, event):
        # Always keep our state and the tracker's state in-sync
        self.tracker.onscroll(event)
        self._cur_histology_transform.transform_atlas_slice_index = self.tracker.ind
        self._cur_histology_transform.transform_atlas_angle = self.tracker.angle

    # Reaction to key pressed
    def on_key(self, event):
        if event.key == '[':
            self.tracker.on_down()
            self._cur_histology_transform.transform_atlas_slice_index = self.tracker.ind
            self._cur_histology_transform.transform_atlas_angle = self.tracker.angle
            return
        elif event.key == ']':
            self.tracker.on_up()
            self._cur_histology_transform.transform_atlas_slice_index = self.tracker.ind
            self._cur_histology_transform.transform_atlas_angle = self.tracker.angle
            return
        if event.key == ';':
            self.tracker.on_angle_up()
            self._cur_histology_transform.transform_atlas_slice_index = self.tracker.ind
            self._cur_histology_transform.transform_atlas_angle = self.tracker.angle
            return
        elif event.key == '\'':
            self.tracker.on_angle_down()
            self._cur_histology_transform.transform_atlas_slice_index = self.tracker.ind
            self._cur_histology_transform.transform_atlas_angle = self.tracker.angle
            return
        # elif event.key == 'u':
            # TODO:??
            # pass
            # try:
            #     self.re_load_images(os.path.splitext(self.names[self.jj])[0])
            # except Exception as e:
            #     self.ERRORS = e
            #     raise
        elif event.key == 't':
            print('\nRegister %s' % self._cur_probe_trace.histology_basename_noext)
            print('Select at least 4 points in the same order in both figures')
            # Call click func on atlas
            self.clicka = 0
            self.cid_atlas = self.fig.canvas.mpl_connect('button_press_event', self.onclick_atlas_hist)
            # self.cid_atlas = self.fig.canvas.mpl_connect('button_press_event', self.onclick_atlas)
            # Call click func on hist
            self.clickh = 0
            # self.cid_hist = self.fig_hist.canvas.mpl_connect('button_press_event', self.onclick_hist)

        elif event.key == 'h':
            print('Transform histology to adapt to the atlas')
            # get the projective transformation from the set of clicked points
            t = transform.ProjectiveTransform()
            t.estimate(np.float32(self._cur_histology_transform.transform_points_atlas),
                       np.float32(self._cur_histology_transform.transform_points_histology))
            img_hist_tempo = np.asanyarray(self.img_hist_temp)
            self._cur_histology_transform.img_warped = transform.warp(img_hist_tempo, t, output_shape=(self.d1, self.d2), order=1, clip=False)  # , mode='constant',cval=float('nan'))
            # Show the  transformed figure
            # fig_trans, ax_trans = plt.subplots(1, 1)#, figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl))
            # =============================================================================
            #         # Select all black pixels
            #         black = np.array([0, 0, 0])
            #         mask = np.abs(img_warped - black).sum(axis=2) < 0.03
            #         img_warped[mask] = [1, 1, 1]
            # =============================================================================
            self.ax_trans.imshow(self._cur_histology_transform.img_warped, origin="lower", extent=[0, self.d1 * self.atlas.pixdim, 0, self.d2 * self.atlas.pixdim])
            self.ax_trans.set_title("Histology adapted to atlas")
            self.fig_trans.canvas.draw()

        elif event.key == 'b':
            print('Simple overlay to scroll through brain regions')
            # SIMPLE OVERLAY
            # here you can scroll the atlas grid
            # self.fig_g, self.ax_g = plt.subplots(1, 1)
            self.ax_g.clear()
            self.fig_g.set_visible(True)
            self.ax_g.imshow(self._cur_histology_transform.img_warped, origin="lower", extent=[0, self.d1 * self.atlas.pixdim, self.d2 * self.atlas.pixdim, 0])
            self.tracker2 = IndexTracker_g(
                ax=self.ax_g,
                pixdim=self.atlas.pixdim,
                p=self.plane,
                S=self.tracker.ind,
                atlas=self.atlas)
            self.fig_g_cid = self.fig_g.canvas.mpl_connect('scroll_event', self.tracker2.onscroll)
            # ax_g.format_coord = format_coord
            self.ax_g.set_title("Histology and atlas overlayed")
            self.fig_g.canvas.draw()
            # Remove axes tick
            self.ax_g.tick_params(axis='both', which='both', bottom=False, left=False, top=False, labelbottom=False,
                            labelleft=False)
        
        elif event.key == 'p':
            print('Overlay to the atlas')
            # get the edges of the colors defined in the label
            # TODO:
            if self.plane == 'c':
                edges = cv2.Canny(np.uint8((self.atlas.cv_plot_plane(y=self.tracker.ind, angle=self.tracker.angle) * 255).transpose((1, 0, 2))), 100, 200)
            elif self.plane == 's':
                edges = cv2.Canny(np.uint8((self.atlas.cv_plot_plane(x=self.tracker.ind, angle=self.tracker.angle) * 255).transpose((1, 0, 2))), 100, 200)
            elif self.plane == 'h':
                edges = cv2.Canny(np.uint8((self.atlas.cv_plot_plane(z=self.tracker.ind, angle=self.tracker.angle) * 255).transpose((1, 0, 2))), 100, 200)
            else:
                raise ValueError(f'Unknown plane {self.plane}')

            # self.fig_grid, self.ax_grid = plt.subplots(1, 1)
            self.ax_grid.clear()
            self.fig_grid.set_visible(True)
            # position of the lines
            self.CC = np.where(edges == 255)
            self._cur_histology_transform.img_warped_overlay = self._cur_histology_transform.img_warped.copy()
            # get the lines in the warped figure
            self._cur_histology_transform.img_warped_overlay[self.CC] = 0.5  # here change grid color (0 black, 1 white)
            self.overlay = self.ax_grid.imshow(self._cur_histology_transform.img_warped_overlay, origin="lower", extent=[0, self.d1 * self.atlas.pixdim, 0, self.d2 * self.atlas.pixdim])
            self.ax_grid.text(0.15, 0.05, self.textstr, transform=self.ax.transAxes, fontsize=6, verticalalignment='bottom', bbox=self.props)
            self.ax_grid.format_coord = self.tracker.format_coord
            self.ax_grid.set_title("Histology and atlas overlayed")
            self.fig_grid.canvas.draw()
            self.cursor = mplcursors.cursor(self.overlay, hover=True)
            self.cursor.connect('add', self.show_annotation)
            self._set_geometry_current_fig(850, 350, self.d2, self.d1)
        
        elif event.key == 'd':
            print('Delete clicked point')
            if self.active == 'atlas':
                try:
                    self._cur_histology_transform.transform_points_atlas.pop(-1)  # remove the point from the list
                    self.rednum_atlas[-1].remove()  # remove the numbers from the plot
                    self.redp_atlas[-1].remove()  # remove the point from the plot
                    self.fig.canvas.draw()
                    self.rednum_atlas.pop(-1)
                    self.redp_atlas.pop(-1)
                    self.clicka -= 1
                except:
                    pass
            else:
                try:
                    self.rednum_hist[-1].remove()  # remove the numbers from the plot
                    self._cur_histology_transform.transform_points_histology.pop(-1)  # remove the point from the list
                    self.redp_hist[-1].remove()  # remove the point from the plot
                    self.fig_hist.canvas.draw()
                    self.redp_hist.pop(-1)
                    self.rednum_hist.pop(-1)
                    self.clickh -= 1
                except:
                    pass
        elif event.key == 'x':
            self._save_histology_transform()
        elif event.key == 'v':
            print('Colored Atlas on')
            # self.fig_color, self.ax_color = plt.subplots(1, 1)
            self.ax_color.clear()
            self.fig_color.set_visible(False)
            self.ax_color.imshow(self._cur_histology_transform.img_warped_overlay, extent=[0, self.d1 * self.atlas.pixdim, 0, self.d2 * self.atlas.pixdim])
            if self.plane == 'c':
                self.ax_color.imshow(self.atlas.cv_plot[:, self.tracker.ind, :].transpose((1, 0, 2)), origin="lower",
                                extent=[0, self.d1 * self.atlas.pixdim, self.d2 * self.atlas.pixdim, 0], alpha=0.5)
            elif self.plane == 's':
                self.ax_color.imshow(self.atlas.cv_plot[self.tracker.ind, :, :].transpose((1, 0, 2)), origin="lower",
                                extent=[0, self.d1 * self.atlas.pixdim, self.d2 * self.atlas.pixdim, 0], alpha=0.5)
            elif self.plane == 'h':
                self.ax_color.imshow(self.atlas.cv_plot[:, :, self.tracker.ind].transpose((1, 0, 2)), origin="lower",
                                extent=[0, self.d1 * self.atlas.pixdim, self.d2 * self.atlas.pixdim, 0], alpha=0.5)
            self.ax_color.set_title("Histology and colored atlas")
            self.fig_color.canvas.draw()

        elif event.key == 'r':
            print('Register probe %d' % (self.probe_counter + 1))
            try:
                self.ax_g.clear()
                self.fig_g.set_visible(False)
                if self.fig_g_cid is not None:
                    self.fig_g.canvas.mpl_disconnect(self.fig_g_cid)
                # plt.close(self.fig_g)
            except:
                pass

            try:
                self.ax_color.clear()
                self.fig_color.set_visible(False)
                # plt.close(self.fig_color)
            except:
                pass

            # Call click func
            self.cid_trans = self.fig_trans.canvas.mpl_connect('button_press_event', self.onclick_probe)
            self.cid_trans2 = self.fig_trans.canvas.mpl_connect('key_press_event', self.on_key2)

        elif event.key == 'w':
            try:
                print('probe %d view mode' % (self.probe_selecter + 1))
                selected_probe = self._probe_traces[self.probe_selecter]
                L = selected_probe.probe_atlas_coords
                probe_x = []
                probe_y = []
                for i in range(len(L)):
                    probe_x.append(L[i][0] * self.atlas.pixdim)
                    probe_y.append(L[i][1] * self.atlas.pixdim)
                # self.fig_probe, self.ax_probe = plt.subplots(1, 1)
                self.ax_probe.clear()
                self.fig_probe.set_visible(False)
                self.trackerp = IndexTracker_p(
                    ax=self.ax_probe,
                    pixdim=self.atlas.pixdim,
                    p=self.plane,
                    S=self.tracker.ind,
                    atlas=self.atlas)
                self.fig_probe.canvas.mpl_connect('scroll_event', self.trackerp.onscroll)
                self.ax_probe.text(0.15, 0.05, self.textstr, transform=self.ax_probe.transAxes, fontsize=6, verticalalignment='bottom', bbox=self.props)
                self.ax_probe.format_coord = self.tracker.format_coord
                self.ax_probe.set_title("Probe viewer")
                self.fig_probe.canvas.draw()
                self.cursor = mplcursors.cursor(self.fig_probe, hover=True)
                self.cursor.connect('add', self.show_annotation)
                self._set_geometry_current_fig(900, 400, self.d2, self.d1)
                # plot the clicked points
                self.ax_probe.scatter(probe_x, probe_y, color=selected_probe.color, s=2)  # , marker='o', markersize=1)
                # plot the probe
                self.pts = np.array((probe_x, probe_y)).T
                # fit the probe
                self.line_fit = Line.best_fit(self.pts)
                if self.line_fit.direction[0] == 0:
                    self.ax_probe.plot(np.array(sorted(probe_x)), np.array(sorted(probe_y)),
                                       color=selected_probe.color, linestyle='dashed', linewidth=0.8)
                else:
                    m, b = np.polyfit(probe_x, probe_y, 1)
                    self.ax_probe.plot(np.array(sorted(probe_x)), m * np.array(sorted(probe_x)) + b,
                                       color=selected_probe.color, linestyle='dashed', linewidth=0.8)
                self.probe_selecter += 1
            except:
                print('No more probes to visualize')
                pass

        elif event.key == 'e':
            self.fig_trans.canvas.mpl_disconnect(self.cid_trans)
            self.fig_trans.canvas.mpl_disconnect(self.cid_trans2)

            self._save_histology_transform()
            self._cur_histology_transform.transform_atlas_angle = self.tracker.angle
            self._cur_histology_transform.transform_atlas_slice_index = self.tracker.ind
            self._cur_probe_trace.histology_transform = self._cur_histology_transform

            cur_probe_trace_fn = os.path.join(
                self.path_probes,
                f'{self._cur_probe_trace.probe_name}_'
                f'{self._cur_probe_trace.histology_basename_noext}.json')
            with open(cur_probe_trace_fn, 'w') as f:
                f.write(self._cur_probe_trace.to_json(indent=4))

            print('Probe points saved')
            
            # Close figures and clear variables
            self.ax_grid.clear()
            self.fig_grid.set_visible(False)
            self.ax_probe.clear()
            self.fig_probe.set_visible(False)

            # Clear out all of the red points
            for arr in (self.rednum_atlas, self.rednum_hist, self.redp_hist, self.redp_atlas):
                for pt in arr:
                    pt.remove()
            self.redp_atlas = []
            self.rednum_atlas = []
            self.redp_hist = []
            self.rednum_hist = []
            self.fig.canvas.draw()
            self.fig_hist.canvas.draw()

            self.p_probe_trans = []
            self.p_probe_grid = []

            self.ax_trans.clear()
            self.fig_trans.canvas.draw()
            self.clicka = 0
            self.clickh = 0

            self._filename_index += 1
            self._initialize_new_file()

            # Disconnnect the registartion of the clicked points to avoid double events
            self.fig.canvas.mpl_disconnect(self.cid_atlas)

            # OPEN A NEW HISTOLOGY FOR NEXT REGISTRATION
            self.plot_hist()

    def _save_histology_transform(self):
        print('\nSave image and slice')
        # Create and save slice, clicked points, and image info
        # Sanity check: sync state between this transform and the tracker in case we've missed it elsewhere
        self._cur_histology_transform.transform_atlas_slice_index = self.tracker.ind
        self._cur_histology_transform.transform_atlas_angle = self.tracker.angle
        bn = self._cur_histology_transform.histology_basename_noext
        fnm = os.path.join(self.path_transformed, bn + '.json')
        with open(fnm, 'w') as f:
            f.write(self._cur_histology_transform.to_json(indent=4))
        # Save the images
        fig_name = bn + '_transformed_withoutlines.jpeg'
        self.fig_trans.savefig(os.path.join(self.path_transformed, fig_name))
        print('\nImage saved')

    



