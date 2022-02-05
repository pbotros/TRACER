
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from PIL import Image, ImageEnhance
from scipy import ndimage
import warnings

warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# Need to keep a global reference to the current processor; else it gets GC'd and doesn't respond to events
_CUR_PROCESSOR = None

def preprocess_histology(histology_folder, file_name=None, plane=None):
    """
    Purpose
    -------------
    Pre-process the histological images, resizing, rotating, changing contrast, etc.
    Each image in the folder should be processed separately.
    Currently only .jpg images can be processed and
    
    Inputs
    -------------
    histology_folder : the path of the folder contains the histological images.

    Outputs
    -------------
    The processed images will be saved in the sub-folder 'processed' in histology_folder.

    """
    if not os.path.exists(histology_folder):
        raise ValueError('Folder path %s does not exist. Please give the correct path.' % histology_folder)

    if file_name is None:
        file_n = input('Histology file name: ')
        if '.jpg' in file_n or '.tif' in file_n:
            file_name = file_n
        else:
            file_name = file_n + '.jpg'

    histology_fn = os.path.join(histology_folder, file_name)
    histology = Image.open(histology_fn).copy()

    # The modified images will be saved in a subfolder called processed
    path_processed = os.path.join(histology_folder, 'processed')
    if not os.path.exists(path_processed):
        os.mkdir(path_processed)
    
    # Insert the plane of interest
    if plane is None:
        plane = input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')
        # Check if the input is correct
        while plane.lower() != 'c' and plane.lower() != 's' and plane.lower() != 'h':
            print('Error: Wrong plane name')
            plane = input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')
    
    #  Size in pixels of the reference atlas brain.
    if plane.lower() == 'c':
        atlas_reference_size = (512, 512)
    elif plane.lower() == 's':
        atlas_reference_size = (512, 1024)
    elif plane.lower() == 'h':
        atlas_reference_size = (1024, 512)
    else:
        raise ValueError()

    if histology.size[0] * histology.size[1] > 89478485:
        oo = histology.size[0] * histology.size[1]
        print('Image size (%d pixels) exceeds limit of 89478485 pixels' % oo)
        print('Image resized to (%d , %d)' % (atlas_reference_size[0], atlas_reference_size[1]))
        histology_width = 1000
        width_percent = (histology_width / float(histology.size[0]))
        height_size = int((float(histology.size[1]) * float(width_percent)))
        histology = histology.resize((histology_width, height_size), Image.NEAREST)

    # # Resize if one of the dimensions is > 2x the atlas'
    print(f'Image is of {histology.size[0]} x {histology.size[1]}...')
    if int(histology.size[0]) > atlas_reference_size[0] * 2 or int(histology.size[1]) > atlas_reference_size[1] * 2:
        ratio0 = float(histology.size[0] / (atlas_reference_size[0] * 2))
        ratio1 = float(histology.size[1] / (atlas_reference_size[1] * 2))
        # Shrink such that at least one dimension is under the size
        ratio = min(ratio0, ratio1)
        print(f'Image is of {histology.size[0]} x {histology.size[1]}: resizing by {ratio}')
        histology = histology.resize(
            (int(histology.size[0] / ratio), int(histology.size[1] / ratio)),
            Image.NEAREST)

    my_dpi = float(histology.info['dpi'][1])

    for key in plt.rcParams.keys():
        if key.startswith('keymap'):
            plt.rcParams[key] = ''

    # Set up figure
    fig = plt.figure(figsize=(float(histology.size[0]) / my_dpi, float(histology.size[1]) / my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(111)
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # =============================================================================
    # ax.set_title("Histology viewer")
    # =============================================================================
    # Show the histology image
    ax.imshow(histology)
    plt.tick_params(labelbottom=False, labelleft=False)

    if histology.size[0] > atlas_reference_size[0] or histology.size[1] > atlas_reference_size[1]:
        print('\nThe image is ' + str(histology.size[0]) + ' x ' + str(histology.size[1]) + ' pixels')
        print('It is recommended to resize this image down to under ' + str(atlas_reference_size[0]) + ' x ' + str(
            atlas_reference_size[1]) + ' pixels\n')
    
    # Downsample and adjust histology image
    # HistologyBrowser(path_processed,histology)
    print('\nControls: \n')
    print('--------------------------- \n')
    print('a: adjust contrast of the image\n')
    print('r: reset to original\n')
    print('f: continue after contrast adjusting\n')
    print('g: Add grid\n')
    print('h: Remove grid\n')
    print('n: set grey scale on\n')
    print('c: crop slice\n')
    print('s: terminate figure editing and save\n')
    print('--------------------------- \n')

    # NB: we want to avoid using both interactive widgets (i.e. matplotlib plots changing in real-time) with keyboard
    # input, as, at least in some Jupyter notebook environments, one blocks the other. Thus we go with the
    # all-Matplotlib approach for both user input and slice display.
    output_fn = os.path.join(path_processed, file_name)

    processor = _HistologyPreprocessor(fig, ax, histology, atlas_reference_size, output_fn)

    # Technically a race condition in case somebody quits before the register happens, but meh
    fig.canvas.mpl_connect('key_press_event', processor.on_key_press)
    fig.canvas.mpl_connect('button_press_event', processor.on_mouse_press)
    plt.show(block=True)


class _HistologyPreprocessor:
    def __init__(self, fig, ax, histology, atlas_reference_size, output_fn):
        self._fig = fig
        self._ax = ax
        self._histology = histology
        self._histology_old = histology.copy()
        self._F = 0
        self._histology_copy = self._histology
        self._process_finished = False

        self._adjusting_in_progress = False
        self._cropping_in_progress = False
        self._cropping_points = []

        self._atlas_reference_size = atlas_reference_size
        self._output_fn = output_fn
        self._line_points_plotted = []

    @property
    def histology(self):
        return self._histology

    def _redraw(self):
        self._ax.clear()
        self._ax.imshow(self._histology)

        if len(self._line_points_plotted) == 1:
            self._ax.scatter(*self._line_points_plotted[0], color='red', marker='o', s=2)
        elif len(self._line_points_plotted) == 2:
            xs = [int(self._line_points_plotted[0][0]), int(self._line_points_plotted[1][0])]
            ys = [int(self._line_points_plotted[0][1]), int(self._line_points_plotted[1][1])]
            self._ax.plot(
                xs,
                ys,
                color='red', marker='o', lw=0.75, markersize=2)
        for pt in self._cropping_points:
            self._ax.scatter(pt[0], pt[1], color='blue', marker='o', s=2)
        self._ax.autoscale_view()
        self._fig.canvas.draw()

    def _adjust_contrast(self, adc):
        enhancer = ImageEnhance.Contrast(self._histology)
        if adc == '+':
            self._histology = enhancer.enhance(1.1)
            self._redraw()
            print('Contrast increased')
        elif adc == '-':
            self._histology = enhancer.enhance(0.9)
            self._redraw()
            print('Contrast decreased')
        elif adc == 'r':
            self._histology = self._histology_copy
            self._redraw()
            print('Original histology restored')
        elif adc == 'f':
            print('Continue editing')
            self._histology_copy = self._histology
            self._adjusting_in_progress = False

    def on_mouse_press(self, event):
        if event.button.name != 'LEFT':
            return

        click_x = event.xdata
        click_y = event.ydata
        if self._cropping_in_progress:
            if len(self._cropping_points) == 4:
                self._cropping_points = []
            else:
                self._cropping_points.append((click_x, click_y))
        else:
            if len(self._line_points_plotted) < 2:
                self._line_points_plotted.append((click_x, click_y))
            else:
                self._line_points_plotted = [(click_x, click_y)]
        self._redraw()

    def on_key_press(self, event):
        self._redraw()

        ipt = event.key
        if self._adjusting_in_progress:
            self._adjust_contrast(ipt)
            return
        if ipt == 'a':
            self._adjusting_in_progress = True
            return

        if ipt == 'g':
            self._redraw()
            myInterval = 50.
            self._ax.xaxis.set_major_locator(plticker.IndexLocator(myInterval, 0))
            self._ax.yaxis.set_major_locator(plticker.IndexLocator(myInterval, 0))
            # Add the grid
            self._ax.grid(which='major', axis='both', linestyle='-')
            # Add the image
            print('Gridd added')
        elif ipt == 'h':
            self._redraw()
            self._ax.grid(False)
            print('Gridd removed')
        elif ipt == 'n':
            self._histology = self._histology.convert('LA')
            self._histology_copy = self._histology  # for a proper rotation
            self._redraw()
            print('Histology in greyscale')
        elif ipt == 'r':
            self._F = 0
            self._histology = self._histology_old  # restore the original histology
            self._histology_copy = self._histology_old  # restore the oiginal histology for rotation
            self._redraw()
            print('Original histology restored')
        elif ipt == 'c':
            # Setting the points for cropped image
            if self._cropping_in_progress:
                if len(self._cropping_points) == 4:
                    left = min([x for x, _ in self._cropping_points])
                    right = max([x for x, _ in self._cropping_points])
                    top = min([y for _, y in self._cropping_points])
                    bottom = max([y for _, y in self._cropping_points])
                    self._histology = self._histology.crop((left, top, right, bottom))
                    self._histology_copy = self._histology  # for a proper rotation
                    print('Cropping done.')
                self._cropping_points = []
                self._cropping_in_progress = False
            else:
                self._cropping_in_progress = True
                print('Cropping now in progress.')

            self._redraw()
        elif ipt == 'o':
            # o for orient
            if len(self._line_points_plotted) != 2:
                return
            xs = [int(self._line_points_plotted[0][0]), int(self._line_points_plotted[1][0])]
            ys = [int(self._line_points_plotted[0][1]), int(self._line_points_plotted[1][1])]
            line_ang = np.rad2deg(np.arctan2([ys[1] - ys[0]], xs[1] - xs[0]))[0] - 90
            self._F += line_ang
            histology_temp = ndimage.rotate(self._histology_copy, self._F, reshape=True)
            self._histology = Image.fromarray(histology_temp)
            self._line_points_plotted = []
            self._redraw()
        elif ipt == 'f':
            self._histology = self._histology.transpose(Image.FLIP_LEFT_RIGHT)
            self._histology_copy = self._histology
            self._redraw()
        elif ipt == 't':
            xlim_range = self._ax.get_xlim()
            ylim_range = self._ax.get_ylim()
            x_dist = xlim_range[1] - xlim_range[0]
            y_dist = ylim_range[1] - ylim_range[0]

            # Resize if one of the dimensions is > 2x the atlas'
            ratio0 = float(self._histology.size[0] / x_dist)
            ratio1 = float(self._histology.size[1] / y_dist)
            ratio = min(ratio0, ratio1)
            print(f'Image is of {self._histology.size[0]} x {self._histology.size[1]}; resizing by {ratio}')
            self._histology = self._histology.resize(
                (int(self._histology.size[0] / ratio), int(self._histology.size[1] / ratio)),
                Image.NEAREST)
            self._histology_copy = self._histology
            self._redraw()
        elif ipt == 's':
            print(self._histology)
            self._histology.save(self._output_fn, compression='lzw')
            print('Histology saved')
