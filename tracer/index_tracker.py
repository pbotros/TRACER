#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:31:04 2020

@author: jacopop
"""


# Class to scroll the atlas slices (coronal, sagittal and horizontal)


class IndexTracker(object):
    def __init__(self, ax, pixdim, p, atlas, S=None, alpha=1.0, edges=False):
        self.ax = ax
        self.plane = p.lower()
        self.ax.set_title('Atlas viewer')
        self._pixdim = pixdim
        self._atlas = atlas
        self._is_edges = edges

        print('\nuse scroll wheel to navigate the atlas \n')

        self._angle = 0
        if self.plane == 'c':
            rows, self.slices, cols = atlas.shape
            if S is None:
                self._ind = atlas.bregma_y_index
            else:
                self._ind = S
            im_kwargs = dict(y=self._ind)
        elif self.plane == 's':
            self.slices, rows, cols = atlas.shape
            if S is None:
                self._ind = 246
            else:
                self._ind = S
            im_kwargs = dict(x=self._ind)
        elif self.plane == 'h':
            rows, cols, self.slices = atlas.shape
            if S is None:
                self._ind = 440
            else:
                self._ind = S
            im_kwargs = dict(z=self._ind)
        else:
            raise ValueError(f'Unknown plane {self.plane}')

        if self._is_edges:
            self._im_getter = atlas.edge_plane
        else:
            self._im_getter = atlas.plane

        im = self._im_getter(**im_kwargs)
        self.im = self.ax.imshow(im.T, origin="lower",
                                 extent=[0, rows * pixdim, 0, cols * pixdim], alpha=alpha,
                                 cmap='gray')  # cmap='gist_yarg' if white background wanted
        self.update()

    @property
    def angle(self):
        return self._angle

    @property
    def ind(self):
        return self._ind

    @angle.setter
    def angle(self, val):
        self._angle = val

    @ind.setter
    def ind(self, val):
        self._ind = val

    def onscroll(self, event):
        if event.button == 'up':
            self.on_up()
        else:
            self.on_down()

    def on_up(self):
        self._ind = (self._ind + 1) % self.slices
        self.update()

    def on_down(self):
        self._ind = (self._ind - 1) % self.slices
        self.update()

    def on_angle_up(self):
        self._angle += 1
        self.update()

    def on_angle_down(self):
        self._angle -= 1
        self.update()

    def format_coord(self, x, y):
        # display the coordinates relative to the bregma when hovering with the cursor
        if self.plane == 'c':
            AP = (self._ind - self._atlas.bregma_y_index) * self._pixdim
            ML = x - 246 * self._pixdim
            Z = y - 440 * self._pixdim
            if ML > 0:
                ML_prefix = 'R'
            else:
                ML_prefix = 'L'
            return f'AP={AP:1.4f}, ML={ML_prefix}{abs(ML):1.4f}, z={Z:1.4f}, angle={self._angle:.1f}'
        elif self.plane == 's':
            AP = x - self._atlas.bregma_y_index * self._pixdim
            ML = self._ind * self._pixdim - 246 * self._pixdim
            Z = y - 440 * self._pixdim
            if ML > 0:
                return f'AP={AP:1.4f}, ML=R{abs(ML):1.4f}, z={Z:1.4f}'
            else:
                return f'AP={AP:1.4f}, ML=L{abs(ML):1.4f}, z={Z:1.4f}'
        elif self.plane == 'h':
            AP = y - self._atlas.bregma_y_index * self._pixdim
            ML = x - 246 * self._pixdim
            Z = self._ind * self._pixdim - 440 * self._pixdim
            if ML > 0:
                return f'AP={AP:1.4f}, ML=R{abs(ML):1.4f}, z={Z:1.4f}'
            else:
                return f'AP={AP:1.4f}, ML=L{abs(ML):1.4f}, z={Z:1.4f}'

    def update(self):
        if self.plane == 'c':
            interped = self._im_getter(y=self._ind, angle=self._angle)
            self.im.set_data(interped.T)
        elif self.plane == 's':
            # Unimplemented
            assert abs(self._angle) < 1e-6
            self.im.set_data(self._im_getter(x=self._ind).T)
        elif self.plane == 'h':
            # Unimplemented
            assert abs(self._angle) < 1e-6
            self.im.set_data(self._im_getter(z=self._ind).T)
        self.ax.set_ylabel('slice %d' % self._ind)
        self.im.axes.figure.canvas.draw()

    def transform_screen_coord(self, xdata, ydata):
        px, py = xdata / self._atlas.pixdim, ydata / self._atlas.pixdim
        if self.plane == 'c':
            if abs(self._angle) < 1e-6:
                return [float(px), float(self._ind), float(py)]
            else:
                x_len = self._atlas.shape[0]
                angle = self._angle
                y_span_half = int(round(np.tan(angle / 180 * np.pi) * x_len / 2))
                y_span = (self._ind - y_span_half, self._ind + y_span_half)
                x_frac = px / x_len
                interped_y = np.interp(x_frac, np.linspace(0, 1, len(y_span)), y_span)
                return [float(px), float(interped_y), float(py)]
        elif self.plane == 's':
            return [float(self._ind), float(px), float(py)]
        elif self.plane == 'h':
            return [float(px), float(py), float(self._ind)]
        else:
            raise ValueError(f'Unknown plane {self.plane}')


class IndexTracker_g(IndexTracker):
    def __init__(self, ax, pixdim, p, S, atlas):
        super(IndexTracker_g, self).__init__(
            ax=ax,
            pixdim=pixdim,
            p=p,
            S=S,
            alpha=0.5,
            atlas=atlas,
            edges=True,
        )


# Class to scroll the atlas slices with probe (coronal, sagittal and horizontal)    
class IndexTracker_p(IndexTracker):
    def __init__(self, ax, pixdim, p, S, atlas):
        super(IndexTracker_p, self).__init__(ax=ax, pixdim=pixdim, p=p, atlas=atlas, S=S, alpha=1.0)


# Class to scroll the boundaries atlas slices (coronal, sagittal and horizontal)
class IndexTracker_b(IndexTracker):
    def __init__(self, ax, pixdim, p, S, atlas):
        super(IndexTracker_b, self).__init__(ax=ax, pixdim=pixdim, p=p, atlas=atlas, S=S, alpha=0.5)


# Class to scroll the color atlas slices (coronal, sagittal and horizontal)            
class IndexTracker_c(IndexTracker):
    def __init__(self, ax, X, pixdim, p, S, atlas):
        plane = p.lower()
        if plane == 'c':
            L = X.transpose((2, 1, 0, 3)).T
        elif plane == 's':
            L = X.transpose((0, 2, 1, 3)).T
        elif self.plane == 'h':
            L = X.transpose((1, 0, 2, 3)).T
        else:
            raise ValueError(f'UNKNOWN plane: {self.plane}')
        super(IndexTracker_c, self).__init__(ax=ax, X=L, pixdim=pixdim, p=p, atlas=atlas, S=S, alpha=0.5)



def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


# Class to scroll the atlas slices with probe (coronal, sagittal and horizontal)
class IndexTracker_pi(object):

    def __init__(self, ax, X, pixdim, p, S, unique_slice, probe_x, probe_y, probe_colors, probe_selecter, line_fit):

        self.ax = ax
        self.plane = p.lower()
        self.unique_slice = unique_slice
        self.probe_x = probe_x
        self.probe_y = probe_y
        self.probe_colors = probe_colors
        self.probe_selecter = probe_selecter
        self.line_fit = line_fit
        ax.set_title('Atlas viewer')
        print('\nuse scroll wheel to navigate the atlas \n')

        self.X = X
        if self.plane == 'c':
            rows, self.slices, cols = X.shape
            self.ind = S
            self.im = ax.imshow(self.X[:, self.ind, :].T, origin="lower", extent=[0, 512 * pixdim, 0, 512 * pixdim],
                                cmap='gray')
            # plot the probe
            if self.line_fit.direction[0] == 0:
                self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),
                                     color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);
            else:
                self.m, self.b = np.polyfit(self.probe_x, self.probe_y, 1)
                self.line = plt.plot(np.array(sorted(self.probe_x)), self.m * np.array(sorted(self.probe_x)) + self.b,
                                     color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);
        elif self.plane == 's':
            self.slices, rows, cols = X.shape
            self.ind = S
            self.im = ax.imshow(self.X[self.ind, :, :].T, origin="lower", extent=[0, 1024 * pixdim, 0, 512 * pixdim],
                                cmap='gray')
            # plot the probe
            if self.line_fit.direction[1] == 0:
                self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),
                                     color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);
            else:
                self.m, self.b = np.polyfit(self.probe_x, self.probe_y, 1)
                self.line = plt.plot(np.array(sorted(self.probe_x)), self.m * np.array(sorted(self.probe_x)) + self.b,
                                     color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);
        elif self.plane == 'h':
            rows, cols, self.slices = X.shape
            self.ind = S
            self.im = ax.imshow(self.X[:, :, self.ind].T, origin="lower", extent=[0, 512 * pixdim, 0, 1024 * pixdim],
                                cmap='gray')
            # plot the probe
            if self.line_fit.direction[0] == 0:
                self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),
                                     color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);
            else:
                self.m, self.b = np.polyfit(self.probe_x, self.probe_y, 1)
                self.line = plt.plot(np.array(sorted(self.probe_x)), self.m * np.array(sorted(self.probe_x)) + self.b,
                                     color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);
        self.points = [plt.scatter(self.probe_x[i], self.probe_y[i], color=self.probe_colors[self.probe_selecter], s=2)
                       for i in range(len(self.probe_x))]
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
            # self.points.remove()
            self.line.pop(0).remove()
            if self.ind not in self.unique_slice:
                self.points = [plt.scatter(self.probe_x[i], self.probe_y[i],
                                           color=lighten_color(self.probe_colors[self.probe_selecter], 0.4), s=2) for i
                               in range(len(self.probe_x))]
                if (self.line_fit.direction[0] == 0 and (self.plane == 'c' or self.plane == 'h')) or (
                        self.line_fit.direction[1] == 0 and self.plane == 's'):
                    self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),
                                         color=lighten_color(self.probe_colors[self.probe_selecter], 0.4),
                                         linestyle='dashed', linewidth=0.8)
                else:
                    self.line = plt.plot(np.array(sorted(self.probe_x)),
                                         self.m * np.array(sorted(self.probe_x)) + self.b,
                                         color=lighten_color(self.probe_colors[self.probe_selecter], 0.4),
                                         linestyle='dashed', linewidth=0.8)
            else:
                self.points = [
                    plt.scatter(self.probe_x[i], self.probe_y[i], color=self.probe_colors[self.probe_selecter], s=2) for
                    i in range(len(self.probe_x))]
                if (self.line_fit.direction[0] == 0 and (self.plane == 'c' or self.plane == 'h')) or (
                        self.line_fit.direction[1] == 0 and self.plane == 's'):
                    self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),
                                         color=self.probe_colors[self.probe_selecter], linestyle='dashed',
                                         linewidth=0.8);
                else:
                    # plot the probe
                    self.line = plt.plot(np.array(sorted(self.probe_x)),
                                         self.m * np.array(sorted(self.probe_x)) + self.b,
                                         color=self.probe_colors[self.probe_selecter], linestyle='dashed',
                                         linewidth=0.8);
        else:
            self.ind = (self.ind - 1) % self.slices
            # self.points.remove()
            self.line.pop(0).remove()
            if self.ind not in self.unique_slice:
                self.points = [plt.scatter(self.probe_x[i], self.probe_y[i],
                                           color=lighten_color(self.probe_colors[self.probe_selecter], 0.4), s=2) for i
                               in range(len(self.probe_x))]
                if (self.line_fit.direction[0] == 0 and (self.plane == 'c' or self.plane == 'h')) or (
                        self.line_fit.direction[1] == 0 and self.plane == 's'):
                    self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),
                                         color=lighten_color(self.probe_colors[self.probe_selecter], 0.4),
                                         linestyle='dashed', linewidth=0.8)
                else:
                    self.line = plt.plot(np.array(sorted(self.probe_x)),
                                         self.m * np.array(sorted(self.probe_x)) + self.b,
                                         color=lighten_color(self.probe_colors[self.probe_selecter], 0.4),
                                         linestyle='dashed', linewidth=0.8)
            else:
                self.points = [
                    plt.scatter(self.probe_x[i], self.probe_y[i], color=self.probe_colors[self.probe_selecter], s=2) for
                    i in range(len(self.probe_x))]
                # plot the probe
                if (self.line_fit.direction[0] == 0 and (self.plane == 'c' or self.plane == 'h')) or (
                        self.line_fit.direction[1] == 0 and self.plane == 's'):
                    self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),
                                         color=self.probe_colors[self.probe_selecter], linestyle='dashed',
                                         linewidth=0.8);
                else:
                    self.line = plt.plot(np.array(sorted(self.probe_x)),
                                         self.m * np.array(sorted(self.probe_x)) + self.b,
                                         color=self.probe_colors[self.probe_selecter], linestyle='dashed',
                                         linewidth=0.8);
        self.update()

    def update(self):
        if self.plane == 'c':
            self.im.set_data(self.X[:, self.ind, :].T)
        elif self.plane == 's':
            self.im.set_data(self.X[self.ind, :, :].T)
        elif self.plane == 'h':
            self.im.set_data(self.X[:, :, self.ind].T)
        self.ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()

    # Class to scroll the atlas slices with probe (coronal, sagittal and horizontal) and color of selected regions


class IndexTracker_pi_col(object):
    def __init__(self, ax, atlas, plane, plane_index, probe_x, probe_y, line_fit, probe_length, lw=1.5, color='black', bg_color=None):
        self.ax = ax
        self.plane = plane.lower()
        self.probe_x = probe_x
        self.probe_y = probe_y
        self.line_fit = line_fit
        self.ind = plane_index

        self._atlas = atlas
        pixdim = atlas.pixdim

        # self.X = X
        # self.edges = edges
        if self.plane == 'c':
            rows, self.slices, cols = atlas.shape
            self.L = atlas.cv_plot_display_plane(y=self.ind)
            self.edges = atlas.edge_plane(y=self.ind)
        elif self.plane == 's':
            self.slices, rows, cols = atlas.shape
            self.L = atlas.cv_plot_display_plane(x=self.ind)
            self.edges = atlas.edge_plane(z=self.ind)
        elif self.plane == 'h':
            rows, cols, self.slices = atlas.shape
            self.L = atlas.cv_plot_display_plane(z=self.ind)
            self.edges = atlas.edge_plane(x=self.ind)
        else:
            raise ValueError(f'Unhandled plane {plane}')

        L_trans = self.L.transpose((1, 0, 2)).copy()
        if bg_color is not None:
            bg_indices = np.argwhere((L_trans.reshape((-1, 3)) == (128, 128, 128)).all(axis=1)).ravel()
            for bg_index in bg_indices:
                i, j = np.unravel_index(bg_index, (L_trans.shape[0], L_trans.shape[1]))
                L_trans[i, j, :] = bg_color
        self.im = ax.imshow(L_trans, origin="lower", extent=[0, rows * pixdim, 0, cols * pixdim])
        print(self.L.transpose((1, 0, 2)).shape)
        # self.im2 = ax.imshow(self.edges, origin="lower", alpha=0.5,
        #                      extent=[0, rows * pixdim, 0, cols * pixdim, ], cmap='gray')
        # plot the probe
        if self.line_fit.direction[0] == 0:
            self.line = ax.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)), linewidth=1.5)
        else:
            first_point = pixdim * line_fit.point
            last_point = pixdim * line_fit.to_point(probe_length)
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
            self.line = ax.plot([first_point[0], last_point[0]], [first_point[1], last_point[1]], color=color, lw=lw)

        self.ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()
