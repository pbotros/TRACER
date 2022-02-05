import pandas as pd
import collections
import math
import os
import pickle
from collections import OrderedDict, Counter

import cv2
import matplotlib.patches as patches
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
# 3d Brain
import vedo
from more_itertools import one
from scipy.spatial import distance
from skspatial.objects import Line, Plane
from tabulate import tabulate

from .ObjSave import probe_obj, ProbeTrace
from .index_tracker import IndexTracker_pi_col


class vis_registered_probes(object):
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

    def __init__(self, atlas, probe_folder=None, plot=True):
        
        if probe_folder is not None:
            if not os.path.exists(probe_folder):
                raise Exception('Please give the correct folder.')

        self.atlas = atlas
        self.probe_folder = probe_folder
        self._should_plot = plot
    
        # PROBE
        self.max_probe_length = 10  # maximum length of probe shank is 10mm
        self.probe_widht = 0.07
        self.probe_thickness = 0.024
        self.probe_tip_length = 0.175
        self.total_electrodes = 960  # total number of recording sites
        self.electrode = 0.012  # Electrode size is 12x12 micron
        self.vert_el_dist = 0.02
        # There are 2 electrodes every 0.02 mm
    
        # # Probe colors
        self.probe_colors = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']
        
        if probe_folder is not None:
            self.set_data_folder(self.probe_folder)

    def set_data_folder(self, probe_folder):
        self.probe_folder = probe_folder
        # The modified images will be saved in a subfolder called processed
        self.path_info = os.path.join(self.probe_folder, 'info')
        if not os.path.exists(self.path_info):
            os.mkdir(self.path_info)
    
        # get the all the files in the probe folder that are not folders
        self.files_probe = []
        for fname in os.listdir(probe_folder):
            if fname.endswith('.json'):
                self.files_probe.append(fname)
        if len(self.files_probe) == 0:
            raise ValueError(f'No probes found in {probe_folder}')

        self.L = probe_obj()
        self.LINE_FIT = probe_obj()
        self.POINTS = probe_obj()
        self.pr = probe_obj()
        self.xyz = probe_obj()
        self.LENGTH = probe_obj()

        self.color_used_t = []

        self._probe_traces = []
        for f in sorted(self.files_probe):
            with open(os.path.join(self.probe_folder, f), 'r') as h:
                trace = ProbeTrace.from_json(h.read())
            self._probe_traces.append(trace)

        def _index_by_probe_name(getter, unique=False):
            ret = collections.defaultdict(list)
            for trace in self._probe_traces:
                ret[trace.probe_name].append(getter(trace))
            if unique:
                return {pn: one(set(values)) for pn, values in ret.items()}
            else:
                return dict(ret)

        # Index a bunch of characteristics of the probe traces for lookup
        probe_traces_by_probe_name = _index_by_probe_name(lambda x: x, unique=False)
        self._probe_planes_by_probe_name = _index_by_probe_name(
            lambda x: x.histology_transform.transform_atlas_plane,
            unique=True)
        self._probe_colors_by_probe_name = _index_by_probe_name(
            lambda x: x.color,
            unique=True)

        # For displaying the right slice at the end
        probe_plane_inds_by_probe_name = _index_by_probe_name(
            lambda x: x.histology_transform.transform_atlas_slice_index,
            unique=False)

        self._best_probe_inds_by_probe_name = {}
        self._probe_names = []

        from sklearn.metrics import pairwise_distances
        for j, (probe_name, probe_traces) in enumerate(probe_traces_by_probe_name.items()):
            self._probe_names.append(probe_name)

            # get the probe coordinates and the region's names
            probe_x = []
            probe_y = []
            probe_z = []
            # Needed to plot colors and probe
            p_x = []
            p_y = []
            planes = []
            max_pt_distances = []
            for probe_trace_idx, probe_trace in enumerate(probe_traces):
                plane = probe_trace.histology_transform.transform_atlas_plane
                planes.append(plane)
                if len(probe_trace.probe_atlas_coords) == 0:
                    max_pt_distances.append(0)
                    continue

                distances = pairwise_distances(np.array(probe_trace.probe_atlas_coords))
                max_pt_distances.append(distances.max())

                for pt in probe_trace.probe_atlas_coords:
                    pt = list(pt)
                    probe_x.append(pt[0])
                    probe_y.append(pt[1])
                    probe_z.append(pt[2])

                    if plane == 'c':
                        p_x.append(pt[0] * self.atlas.pixdim)
                        p_y.append(pt[2] * self.atlas.pixdim)
                    elif plane == 's':
                        p_x.append(pt[1] * self.atlas.pixdim)
                        p_y.append(pt[2] * self.atlas.pixdim)
                    elif plane == 'h':
                        p_x.append(pt[0] * self.atlas.pixdim)
                        p_y.append(pt[1] * self.atlas.pixdim)
                    else:
                        raise ValueError(f'Unhandled plane {plane}')

            self._best_probe_inds_by_probe_name[probe_name] = \
                probe_traces[np.argmax(max_pt_distances)].histology_transform.transform_atlas_slice_index
            self.pts = np.array((probe_x, probe_y, probe_z)).T
            print(f'j={j}: probe_name={probe_name}; {len(self.pts)} pts for probe across {len(probe_traces)} traces')
            # fit the probe

            line_fit = Line.best_fit(self.pts)

            pts_projected = [line_fit.project_point(pt) for pt in self.pts]

            distances = pairwise_distances(pts_projected)
            farthest_i, farthest_j = np.unravel_index(np.argmax(distances.ravel()), distances.shape)
            first_point = self.pts[farthest_i, :]
            last_point = self.pts[farthest_j, :]

            # Thinking about coronal sections for plotting, take the "first point" as the one highest up in
            # the brain, which is then flipped in Z coordinates
            if last_point[2] < first_point[2]:
                first_point, last_point = last_point, first_point

            new_direction = line_fit.project_point(last_point) - line_fit.project_point(first_point)
            new_direction /= np.linalg.norm(new_direction, 2)

            # Ensure the line starts at the "first" point
            line_fit = Line(point=line_fit.project_point(first_point), direction=new_direction)

            # Adjust the line's "last_point" to be right where it inserts into cortex
            _, _, final_t, _ = self._determine_insertion_point(line_fit=line_fit)
            last_point = line_fit.to_point(final_t)

            # if no inclination in z direction
            if line_fit.direction[2] == 0:
                # line equations, to derive the starting and end point of the line (aka probe)
                z1 = first_point[2]
                x1 = first_point[0]
                y1 = line_fit.point[1]+((x1-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                z2 = first_point[2]
                x2 = last_point[0]
                y2 = line_fit.point[1]+((x2-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
            else:
                x1, y1, z1 = line_fit.project_point(first_point)
                x2, y2, z2 = line_fit.project_point(last_point)

                probe_length = np.linalg.norm(np.array([x2 - x1, y2 - y1, z2 - z1]), 2)
                probe_length_minus_tip = probe_length - self.probe_tip_length

                xt = np.average([x1, x2], weights=[probe_length_minus_tip, 1 - probe_length_minus_tip])
                yt = np.average([y1, y2], weights=[probe_length_minus_tip, 1 - probe_length_minus_tip])
                zt = np.average([z1, z2], weights=[probe_length_minus_tip, 1 - probe_length_minus_tip])

            dist = distance.euclidean((x1, y1, z1), (x2, y2, z2)) # probe length
            dist_mm = dist * self.atlas.pixdim  # probe length in mm
            # get the line to plot
            probe_color = self._probe_colors_by_probe_name[probe_name]
            l = vedo.Line([x1, y1, z1], [x2, y2, z2], c=probe_color, lw=2)
            # clicked points to display
            pp = vedo.Points(self.pts, c=probe_color)  # fast
            setattr(self.xyz, probe_color, [[x1, y1, z1], [xt, yt, zt]])
            setattr(self.LENGTH, probe_color, [dist_mm, dist])
            setattr(self.pr, probe_color, pp)
            setattr(self.L, probe_color, l)
            setattr(self.LINE_FIT, probe_color, line_fit)
            setattr(self.POINTS, probe_color, [p_x, p_y])
            self.color_used_t.append(probe_color)

        # get only the unique color in order
        self.color_used = list(OrderedDict.fromkeys(self.color_used_t))
        self.n = len(self.color_used)
        print(('n', self.n))

    def vis2d(self):
        # Ed = np.load(path_files/'Edges.npy')
        # To plot the probe with colors
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        # compute and display the insertion angle for each probe
        for i in range(0, self.n):
            probe_name = self._probe_names[i]
            line_fit = getattr(self.LINE_FIT, self.color_used[i])
            deg_lat = math.degrees(math.atan(line_fit.direction[0]))
            deg_ant = math.degrees(math.atan(line_fit.direction[1]))
            Length = getattr(self.LENGTH, self.color_used[i])
            print('\n\nAnalyze %s probe: \n ' % self.color_used[i])
            print('Probe length: %.2f mm \n' % Length[0])
            print('Estimated %s probe insertion angle: ' % self.color_used[i])
            print('%.2f degrees in the anterior direction' % deg_ant)
            print('%.2f degrees in the lateral direction\n' % deg_lat)
            
            # Get the brain regions traversed by the probe
            X1 = getattr(self.xyz, self.color_used[i])[0]
            X2 = getattr(self.xyz, self.color_used[i])[1]
        
            regions = []
            colori = []
            initials = []
            index = []
            channels = []
            point_along_line = []

            from tqdm import tqdm
            print('Processing segmentation regions.')
            for t in tqdm(np.linspace(0, Length[1], 300)[::-1]):
                x, y, z = line_fit.to_point(t)
                point_along_line.append([x, y, z])
                region = self.atlas.segmentation_region(x=int(x), y=int(y), z=int(z))
                if region != None:
                    regions.append(region['name'])
                    colori.append(region['color'])
                    initials.append(region['initial'])

            if Length[0] > self.max_probe_length + self.probe_tip_length:
                print('ERROR: probe %d (%s) exceed the maximum probe length (10mm)!\n' % (i+1, self.color_used[i]))

            # count the number of elements in each region to
            # TODO: have this go sequentially...
            # regioni = list(OrderedDict.fromkeys(regions))
            # iniziali = list(OrderedDict.fromkeys(initials))
            # # remove clear label
            # if 'Clear Label' in regioni:
            #     indice = regioni.index('Clear Label')
            #     regioni.pop(indice)
            #     iniziali.pop(indice)
            # cc = 0
            # jj = 0

            region_streaks = []

            cur_region = None
            num_in_cur_region = 0
            for t_idx, region_name in enumerate(regions):
                if cur_region is None:
                    cur_region = region_name
                    num_in_cur_region = 0

                if cur_region == region_name:
                    num_in_cur_region += 1
                else:
                    region_streaks.append((cur_region, t_idx - num_in_cur_region, t_idx))
                    cur_region = region_name
                    num_in_cur_region = 1
            if cur_region is not None:
                region_streaks.append((cur_region, len(regions) - num_in_cur_region, len(regions)))

            # num_el = []
            # indici = []

            region_name_counter = collections.defaultdict(lambda: 0)

            region_stats = []
            for region_name, region_start_idx, region_end_idx in region_streaks:
                region_count = region_name_counter[region_name]
                region_name_counter[region_name] += 1

                if region_count == 0:
                    region_name_with_count = region_name
                else:
                    region_name_with_count = f'{region_name} ({region_count + 1})'

                if region_start_idx != region_end_idx:
                    regional_dist = distance.euclidean(
                        np.array(point_along_line[region_start_idx]) * self.atlas.pixdim,
                        np.array(point_along_line[region_end_idx - 1]) * self.atlas.pixdim,
                    )
                else:
                    regional_dist = self.atlas.pixdim

                # Number of electrodes in the region
                # num_el.append(round(regional_dist/self.vert_el_dist)*2)
                if regions[region_start_idx] != region_name:
                    raise ValueError(f'actual={regions[region_start_idx]}, expected={region_name}')
                region_stats.append({
                    'Channels': round(regional_dist/self.vert_el_dist)*2,
                    ' Regions traversed': region_name_with_count,
                    'Initials': initials[region_start_idx],
                })
                # proportion of the probe in the given region
                # dist_prop = Length[1]/regional_dist
                color_prop = self.atlas.labels_color[np.argwhere(np.array(self.atlas.labels_name) == region_name)]
                # length of the longest probe
                # m = []
                # for k in range(0, self.n):
                #     mt = getattr(self.LENGTH, self.color_used[k])
                #     m.append(mt[1])
                # max_val = max(m)
                # plot the probe with the colors of the region traversed
                # ax1.add_patch(patches.Rectangle((100*i+20, cc), 17, dist_prop, color=color_prop[0][0]/255))
                # ax1.text(100*i, (max_val + 10), 'Probe %d\n(%s)' % (i+1, self.color_used[i]), color=self.color_used[i], fontsize=9, fontweight='bold')
                # if len(iniziali[jj]) > 7:
                #     ax1.text(100*i-12, cc+2, '%s-\n%s' % (iniziali[jj][0:5], iniziali[jj][6:]), fontsize=5.6)
                # else:
                #     ax1.text(100*i-12, cc+4, '%s' % (iniziali[jj]), fontsize=6)
                # ax1.text(100*i+48, cc+4, '%d' % (num_el[jj]), fontsize=6.5)
                # jj += 1
                # cc = dist_prop + cc
            # for re in regioni:
            #     # store the index o the region to print only the color of the regions of interest
            #     region_index = self.atlas.labels_name.index(re)
            #     indici.append(region_index)
            #     # in the case in dont exit and then enter again the region
            #     start_i = None
            #     end_i = None
            #     for region_index, region_name in enumerate(regions):
            #         if region_name == re:
            #             if end_i is not None:
            #                 print(f'==== WARNING: Re-entered region? {re}')
            #                 break
            #             if start_i is None:
            #                 start_i = region_index
            #         elif region_name != re and start_i is not None:
            #             end_i = region_index
            #
            #     position = [region_index for region_index, region_name in enumerate(regions) if region_name == re]
            #     # if there is only one point in the region
            #     if len(position) == 1:
            #         regional_dist = self.atlas.pixdim
            #     else:
            #         # first point along the line in the region
            #         start = [element * self.atlas.pixdim for element in point_along_line[position[0]]]
            #         # last point along the line in the region
            #         end = [element * self.atlas.pixdim for element in point_along_line[position[-1]]]
            #         # length of the part of the probe in the region
            #         regional_dist = distance.euclidean(start,end)
            #     # Number of electrodes in the region
            #     num_el.append(round(regional_dist/self.vert_el_dist)*2)
            #     #print(re)
            #     # proportion of the probe in the given region
            #     dist_prop = Length[1]/len(regioni)
            #     color_prop = self.atlas.labels_color[np.argwhere(np.array(self.atlas.labels_name) == re)]
            #     # length of the longest probe
            #     # plot the probe with the colors of the region traversed
            #     ax1.add_patch(patches.Rectangle((100*i+20, cc), 17, dist_prop, color=color_prop[0][0]/255))
            #     ax1.text(100*i, (max_val + 10), 'Probe %d\n(%s)' % (i+1, self.color_used[i]), color=self.color_used[i], fontsize=9, fontweight='bold')
            #     if len(iniziali[jj]) > 7:
            #         ax1.text(100*i-12, cc+2, '%s-\n%s' % (iniziali[jj][0:5], iniziali[jj][6:]), fontsize=5.6)
            #     else:
            #         ax1.text(100*i-12, cc+4, '%s' % (iniziali[jj]), fontsize=6)
            #     ax1.text(100*i+48, cc+4, '%d' % (num_el[jj]), fontsize=6.5)
            #     jj += 1
            #     cc = dist_prop + cc
            #     del regional_dist, position
                
            # LL = [regioni,  iniziali, num_el]
            # headers = [' Regions traversed', 'Initials', 'Channels']
            # numpy_array = np.array(LL)
            # transpose = numpy_array.T
            # transpose_list = transpose.tolist()
            # transpose_list.reverse()
            # print(tabulate(transpose_list, headers, floatfmt=".2f"))
            region_stats = pd.DataFrame(region_stats)
            print(tabulate(region_stats, headers=region_stats.columns, floatfmt=".2f", showindex=False))
            punti = getattr(self.POINTS, self.color_used[i])

            # TODO: for only plotting the colors we care about?
            # cv_plot_display = np.load(path_files/'cv_plot_display.npy')
            # for j in range(len(self.atlas.labels_index)):
            #     if j in indici:
            #         coord = np.where(self.atlas.segmentation_data == self.atlas.labels_index[j][0])
            #         self.atlas.cv_plot_display[coord[0],coord[1],coord[2],:] = self.atlas.labels_color[j]
             #fig_color_probe, axs_color_probe = plt.subplots(1, 3) # to plot the region interested with colors
            fig_color_probe, ax_color_probe = plt.subplots() # to plot the region interested with colors
            # for ax_color_probe, plane in zip(axs_color_probe, ('c', 'h', 's')):
            IndexTracker_pi_col(
                ax=ax_color_probe,
                atlas=self.atlas,
                # self.atlas.cv_plot_display / 255,
                # self.atlas.Edges,
                # self.atlas.pixdim,
                plane=self._probe_planes_by_probe_name[probe_name],
                plane_index=self._best_probe_inds_by_probe_name[probe_name],
                probe_length=Length[1],
                probe_x=punti[0],
                probe_y=punti[1],
                line_fit=line_fit)
            ax_color_probe.set_title('Probe %d\n(%s)' % (i+1, self.color_used[i]))

            fig_out = os.path.join(self.path_info, f'Probe_{self.color_used[i]}.pdf')
            fig_color_probe.savefig(fig_out, bbox_inches='tight')

            if self._should_plot:
                plt.show()
            else:
                plt.close(fig1)
                plt.close(fig_color_probe)

            vs_depths = [5.5, 6.0]
            insertion_ap, insertion_ml, final_t, vs_lengths = self._determine_insertion_point(
                probe_i=i, vs_depths=vs_depths)

            # Write and save txt file with probe info
            pn = "Probe_%s.txt" % self.color_used[i]
            f = open(os.path.join(self.path_info, pn),"w+")
            f.write('Analyze probe: \n\n ')
            f.write('Probe length: %.2f mm \n\n' % Length[0])
            f.write('Estimated probe insertion angle:  \n')
            f.write('%.2f degrees in the anterior direction \n' % deg_ant)
            f.write('%.2f degrees in the lateral direction\n\n' % deg_lat)
            f.write(f'Insertion point into cortex: AP={insertion_ap:1.4f}, ML={abs(insertion_ml):1.4f}\n')
            f.write(f'Probe length from insertion point to ventral striatum:\n')
            for vs_depth, vs_length in zip(vs_depths, vs_lengths):
                f.write(f'=> VS @ {vs_depth}: {vs_length}\n')
            f.write('\n')
            f.write(tabulate(region_stats, headers=region_stats.columns, floatfmt=".2f", showindex=False))
            f.close()

            with open(os.path.join(self.path_info, pn), 'r') as f:
                print(f.read())

        
        ax1.axis(xmin=0,xmax=100*self.n+20)
        m = []
        for k in range(0, self.n):
            mt = getattr(self.LENGTH, self.color_used[k])
            m.append(mt[1])
        max_val = max(m)
        ax1.axis(ymin=0,ymax=max_val)
        ax1.axis('off')

    def _determine_insertion_point(self, probe_i=None, line_fit=None, vs_depths=None):
        print(f'Determining insertion point for probe {probe_i}')
        if line_fit is None:
            assert probe_i is not None
            line_fit = getattr(self.LINE_FIT, self.color_used[probe_i])
        else:
            assert probe_i is None
        # Go 7mm in one direction
        probe_length_max = 7 / self.atlas.pixdim

        upper_z_plane = Plane(point=[0, 0, self.atlas.shape[2] - 1], normal=[0, 0, 1])
        start_t = line_fit.transform_points(upper_z_plane.intersect_line(line_fit))[0]
        end_t = probe_length_max / 2

        x, y, z = line_fit.to_point(start_t)
        region = self.atlas.segmentation_region(x=int(x), y=int(y), z=int(z))['name']
        assert region == 'Clear Label'

        ts = np.linspace(start_t, end_t, 2000)

        # Binary search to find the point where the regions switch from "Clear Label"
        # to cortex
        upper_idx = len(ts)
        lower_idx = 0
        final_t = None
        while True:
            idx = (lower_idx + upper_idx)//2

            if abs(upper_idx - lower_idx) <= 1:
                regions = []
                for idx_cand in (idx - 2, idx - 1, idx, idx + 1):
                    assert idx_cand >= 0
                    t = ts[idx_cand]
                    x, y, z = line_fit.to_point(t)
                    r = self.atlas.segmentation_region(x=int(x), y=int(y), z=int(z))
                    assert r is not None
                    regions.append(r['name'])

                if regions[0] == 'Clear Label' and regions[1] in ('neocortex', 'Primary motor area'):
                    final_t = ts[idx - 1]
                elif regions[1] == 'Clear Label' and regions[2] in ('neocortex', 'Primary motor area'):
                    final_t = ts[idx]
                elif regions[2] == 'Clear Label' and regions[3] in ('neocortex', 'Primary motor area'):
                    final_t = ts[idx + 1]
                else:
                    raise ValueError('Huh??')
                break

            t = ts[idx]
            x, y, z = line_fit.to_point(t)
            region = self.atlas.segmentation_region(x=int(x), y=int(y), z=int(z))['name']
            if region == 'Clear Label':
                lower_idx = idx + 1
            else:
                upper_idx = idx

        assert final_t is not None
        final_t = float(final_t)
        x_insertion, y_insertion, z_insertion = line_fit.to_point(final_t)
        print('Insertion point in pixels: ', x_insertion, y_insertion, z_insertion)

        pixdim = self.atlas.pixdim
        AP = (y_insertion - self.atlas.bregma_y_index) * pixdim
        ML = (x_insertion - self.atlas.bregma_x_index) * pixdim
        # Z = y_insertion - 440 * pixdim

        # Where does the line intersect our defined "ventral striatum"
        if vs_depths  is not None:
            lengths_vs = []
            for vs_depth in vs_depths:
                vs_depth_z = z_insertion + vs_depth / pixdim
                vs_depth_pixels = vs_depth / pixdim
                vs_plane = Plane(point=[0, 0, vs_depth_z], normal=[0, 0, 1])
                vs_t = line_fit.transform_points(vs_plane.intersect_line(line_fit))[0]
                length_vs = abs(vs_t - final_t) * pixdim
                lengths_vs.append(length_vs)
        else:
            lengths_vs = []
        return AP, ML, final_t, lengths_vs

    def vis2d_all(self, lw=1.0, bg_color=None):

        # All probes should be oriented in the same plane
        for plane in ('c', 's', 'h',):
            plane_indices = []
            for i in range(0, self.n):
                line_fit = getattr(self.LINE_FIT, self.color_used[i])
                if plane == 'c':
                    plane_index_index = 1
                elif plane == 's':
                    plane_index_index = 0
                elif plane == 'h':
                    plane_index_index = 2
                else:
                    raise ValueError(f'Unhandled plane {plane}')
                plane_indices.append(line_fit.point[plane_index_index])

            plane_index = int(np.median(plane_indices))
            fig_color_probe, ax_color_probe = plt.subplots() # to plot the region interested with colors

            for i in range(0, self.n):
                probe_name = self._probe_names[i]
                line_fit = getattr(self.LINE_FIT, self.color_used[i])
                Length = getattr(self.LENGTH, self.color_used[i])
                punti = getattr(self.POINTS, self.color_used[i])

                probe_length = Length[1]
                pixdim = self.atlas.pixdim
                first_point = pixdim * line_fit.point
                last_point = pixdim * line_fit.to_point(probe_length)

                if i == 0:
                    IndexTracker_pi_col(
                        ax=ax_color_probe,
                        atlas=self.atlas,
                        # self.atlas.cv_plot_display / 255,
                        # self.atlas.Edges,
                        # self.atlas.pixdim,
                        plane=plane,
                        plane_index=plane_index,
                        # plane=self._probe_planes_by_probe_name[probe_name],
                        # plane_index=self._best_probe_inds_by_probe_name[probe_name],
                        probe_length=Length[1],
                        probe_x=punti[0],
                        probe_y=punti[1],
                        line_fit=line_fit,
                        lw=lw,
                        bg_color=bg_color)
                else:
                    # plot the probe
                    if plane == 'c':
                        first_point = [first_point[0], first_point[2]]
                        last_point = [last_point[0], last_point[2]]
                    elif plane == 'h':
                        first_point = [first_point[0], first_point[1]]
                        last_point = [last_point[0], last_point[1]]
                    elif plane == 's':
                        first_point = [first_point[1], first_point[2]]
                        last_point = [last_point[1], last_point[2]]
                    else:
                        raise ValueError(f'Unhandled plane {plane}')

                    xx = [first_point[0], last_point[0]]
                    yy = [first_point[1], last_point[1]]
                    print(f'{plane}: Plotting {xx}, {yy}')
                    ax_color_probe.plot(
                        xx,
                        yy,
                        color='black',
                        lw=lw)

            ax_color_probe.grid(False)
            fig_color_probe.savefig(os.path.join(self.path_info, f'all_probes_{plane}.pdf'), bbox_inches='tight')

    def vis3d(self):
        vedo.settings.embedWindow(backend=False, verbose=True)
        # load the brain regions
        # mask_data = self.atlas.mask_data.transpose((2, 1, 0))
        # Edges = np.empty((512, 1024, 512))
        # for sl in range(0, 1024):
        #     Edges[:, sl, :] = cv2.Canny(np.uint8((mask_data[:, sl, :]) * 255), 100, 200)
        #
        coords = self.atlas.edges_coords()
        # Manage Points cloud
        points = vedo.pointcloud.Points(coords)
        # Create the mesh
        mesh = vedo.mesh.Mesh(points)
        print(2)
        # create some dummy data array to be associated to points
        data = mesh.points()[:, 2]  # pick z-coords, use them as scalar data
        # build a custom LookUp Table of colors:
        lut = vedo.buildLUT([
            (512, 'white', 0.07),
        ],
            vmin=0, belowColor='lightblue',
            vmax=512, aboveColor='grey',
            nanColor='red',
            interpolate=False,
        )
        mesh.cmap(lut, data)

        # plot all the probes together
        p3d = vedo.Plotter(title='Brain viewer', size=(700, 700), pos=(250, 0))
        vedo_points0 = getattr(self.pr, self.color_used[0])
        if self.n == 1:
             p3d += [vedo.fitLine(vedo_points0).lw(4).c('black')]
             p3d.show(mesh, vedo_points0, getattr(self.L, self.color_used[0]), __doc__,
                      axes=0, viewup="z", bg='black',
                      )
        elif self.n == 2:
             p3d.show(mesh, vedo_points0, getattr(self.pr, self.color_used[1]), getattr(self.L, self.color_used[0]), getattr(self.L, self.color_used[1]), __doc__,
                      axes=0, viewup="z", bg='black',
                      )
        elif self.n == 3:
             p3d.show(mesh, vedo_points0, getattr(self.pr, self.color_used[1]), getattr(self.pr, self.color_used[2]), getattr(self.L, self.color_used[0]), getattr(self.L, self.color_used[1]), getattr(self.L, self.color_used[2]), __doc__,
                      axes=0, viewup="z", bg='black',
                      )
        elif self.n == 4:
             p3d.show(mesh, vedo_points0, getattr(self.pr, self.color_used[1]), getattr(self.pr, self.color_used[2]), getattr(self.pr, self.color_used[3]), getattr(self.L, self.color_used[0]), getattr(self.L, self.color_used[1]), getattr(self.L, self.color_used[2]), getattr(self.L, self.color_used[3]), __doc__,
                      axes=0, viewup="z", bg='black',
                      )
        elif self.n == 5:
             p3d.show(mesh, vedo_points0, getattr(self.pr, self.color_used[1]), getattr(self.pr, self.color_used[2]), getattr(self.pr, self.color_used[3]), getattr(self.pr, self.color_used[4]), getattr(self.L, self.color_used[0]), getattr(self.L, self.color_used[1]), getattr(self.L, self.color_used[2]), getattr(self.L, self.color_used[3]), getattr(self.L, self.color_used[4]), __doc__,
                      axes=0, viewup="z", bg='black',
                      )
        elif self.n == 6:
             p3d.show(mesh, vedo_points0, getattr(self.pr, self.color_used[1]), getattr(self.pr, self.color_used[2]), getattr(self.pr, self.color_used[3]), getattr(self.pr, self.color_used[4]), getattr(self.pr, self.color_used[5]), getattr(self.L, self.color_used[0]), getattr(self.L, self.color_used[1]), getattr(self.L, self.color_used[2]), getattr(self.L, self.color_used[3]), getattr(self.L, self.color_used[4]), getattr(self.L, self.color_used[5]), __doc__,
                      axes=0, viewup="z", bg='black',
                      )
        # p3d.export(os.path.join(self.path_info, 'probe_3d.x3d'), binary=False)

