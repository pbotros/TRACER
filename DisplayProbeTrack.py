from __future__ import print_function

# Import libraries
import math 
import os
import os.path
import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import pickle 
from skimage import measure
from collections import OrderedDict
    
# 3d Brain
from vedo import Volume as VedoVolume
import vedo 
from vedo import buildLUT, Sphere, show, settings
from vedo.mesh import Mesh, merge
from vedo import *

# fit the probe
from skspatial.objects import Line
#from skspatial.plotting import plot_3d

from ObjSave import  probe_obj, save_probe

# read label file
from Readlabel import readlabel

# Labels
# Mac
labels_item = open(r"/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.label", "r")
# Windows
# labels_item = open(r"C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas\WHS_SD_rat_atlas_v4_beta.label", "r")
labels_index, labels_name, labels_color = readlabel( labels_item )   

# Segmentation
# Mac
segmentation = nib.load('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.nii.gz')
# Windows
#segmentation = nib.load(segmentation_path)
segmentation_data = segmentation.get_fdata()

# Probe colors
probe_colors = ['green', 'purple', 'blue', 'yellow', 'orange', 'red']

# Windows
# =============================================================================
# processed_histology_folder = r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\histology\processed'
# path_probes = os.path.join(processed_histology_folder, 'probes')
# path_transformed = os.path.join(processed_histology_folder, 'transformations')
# =============================================================================

# Mac
processed_histology_folder = r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed'
path_probes = r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes'
path_transformed = '/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/transformations'

# get the all the files in the probe folder
files_probe = os.listdir(path_probes)
files_transformed = os.listdir(path_transformed)

L = probe_obj()
LINE_FIT = probe_obj() 
pr = probe_obj()
xyz = probe_obj()
P = []
color_used_t = []
# =============================================================================
# for f in files_probe:
#     # WINDOWS
#     P.append(pickle.load(open(os.path.join(path_probes, f), "rb")))
# =============================================================================
    # MAC
P.append(pickle.load(open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes/1probes.pkl', "rb")))
P.append(pickle.load(open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes/2probes.pkl', "rb")))
# LL = pickle.load(open(os.path.join(path_probes, '1probes.pkl'), "rb"))    
probe_counter = P[0].Counter

# If I have several probes
for j in range(len(probe_colors)):    
    # get the probe coordinates and the region's names
    probe_x = []
    probe_y = []
    probe_z = []
    for k in range(len(P)):
        try:
            PC = getattr(P[k].Probe, probe_colors[j])
            if P[k].Plane == 'c':
                for i in range(len(PC)):
                    probe_x.append(PC[i][0])
                    probe_y.append(P[k].Slice)
                    probe_z.append(PC[i][1])
            elif P[k].Plane == 'c':
                for i in range(len(PC)):
                    probe_x.append(P[k].Slice)
                    probe_y.append(PC[i][0])
                    probe_z.append(PC[i][1])  
            elif P[k].Plane == 'c':
                for i in range(len(PC)):        
                    probe_x.append(PC[i][0])
                    probe_y.append(PC[i][1])        
                    probe_z.append(P[k].Slice)
            pts = np.array((probe_x, probe_y, probe_z)).T

            # fit the probe
            line_fit = Line.best_fit(pts)
            # line equations, to derive the starting and end point of the line (aka probe)
            z1 = max(pts[:,2])
            x1 = line_fit.point[0]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
            y1 = line_fit.point[1]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
            z2 = min(pts[:,2])
            x2 = line_fit.point[0]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
            y2= line_fit.point[1]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
            # get the line to plot
            l = vedo.Line([x1, y1, z1],[x2, y2, z2],c = probe_colors[j], lw = 2)
            # clicked points to display
            pp = vedo.Points(pts, c = probe_colors[j]) #fast    
            setattr(xyz,probe_colors[j], [[x1, y1, z1], [x2, y2, z2]])
            setattr(pr, probe_colors[j], pp)
            setattr(L, probe_colors[j], l)
            setattr(LINE_FIT, probe_colors[j], line_fit)
            color_used_t.append(probe_colors[j])
        except:
            pass  

# get only the unique color in order                  
color_used = list(OrderedDict.fromkeys(color_used_t))
n = len(color_used)

# load the brain regions
Edges = np.load('Edges.npy')
edges = Edges.T
coords = np.array(np.where(edges == 255))
# Manage Points cloud
points = vedo.pointcloud.Points(coords)
# Create the mesh
mesh = Mesh(points)
# create some dummy data array to be associated to points
data = mesh.points()[:,2]  # pick z-coords, use them as scalar data
# build a custom LookUp Table of colors:
lut = buildLUT([
                (512, 'lightgrey', 0.01 ),
               ],
               vmin=0, belowColor='lightblue',
               vmax= 512, aboveColor='grey',
               nanColor='red',
               interpolate=False,
              )
mesh.cmap(lut, data)


regions = []
# compute and display the insertion angle for each probe
for i in range(0,n):
    line_fit = getattr(LINE_FIT, color_used[i])
    deg_lat = math.degrees(math.atan(line_fit.direction[0]))
    deg_ant = math.degrees(math.atan(line_fit.direction[1]))
    print('\nestimated %s probe insertion angle: ' %color_used[i])
    print('%.2f degrees in the anterior direction' %deg_ant)
    print('%.2f degrees in the lateral direction' %deg_lat)

    # Get the brain regions traversed by the probe
    X1 = getattr(xyz, color_used[i])[0]
    X2 = getattr(xyz, color_used[i])[1]
    s = min([int(math.modf(X1[2])[1]),int(math.modf(X2[2])[1])]) # starting point
    f = max([int(math.modf(X1[2])[1]),int(math.modf(X2[2])[1])]) # ending point
    for z in range(s,f):
        x = line_fit.point[0]+((z-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
        y = line_fit.point[1]+((z-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
        regions.append(labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
    regioni = list(OrderedDict.fromkeys(regions))  
    print('\nRegions traversed by %s probe\n: ' %color_used[i])
    for re in regioni:
        print(re)
        




# plot all the probes together
if n==1:
     show(mesh, getattr(pr,color_used[0]), getattr(L, color_used[0]), __doc__,
     axes=dict(zLabelSize=.04, numberOfDivisions=10),
     elevation=-80, bg='white',
     )        
elif n==2:     
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(L, color_used[0]), getattr(L, color_used[1]), __doc__,
     axes=dict(zLabelSize=.04, numberOfDivisions=10),
     elevation=-80, bg='white',
     )
elif n == 3:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(L, color_used[0]),getattr(L, color_used[1]), getattr(L, color_used[2]), __doc__,
     axes=dict(zLabelSize=.04, numberOfDivisions=10),
     elevation=-80, bg='white',
     )
elif n == 4:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), __doc__,
     axes=dict(zLabelSize=.04, numberOfDivisions=10),
     elevation=-80, bg='white',
     )
elif n==5:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), getattr(L, color_used[4]),  __doc__,
     axes=dict(zLabelSize=.04, numberOfDivisions=10),
     elevation=-80, bg='white',
     )
elif n == 6:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(pr,color_used[5]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), getattr(L, color_used[4]), getattr(L, color_used[5]), __doc__,
     axes=dict(zLabelSize=.04, numberOfDivisions=10),
     elevation=-80, bg='white',
     )















