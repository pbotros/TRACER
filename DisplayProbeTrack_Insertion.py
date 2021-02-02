"""Brain viewer"""
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
import matplotlib.patches as patches
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import pickle 
from skimage import measure
from collections import OrderedDict, Counter
from tabulate import tabulate
from scipy.spatial import distance    

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

# PROBE
max_probe_length = 10 # maximum length of probe shank is 10mm
probe_widht = 0.07  
probe_thickness = 0.024
probe_tip_length = 0.175  
total_electrodes = 960 # total number of recording sites
electrode = 0.012 # Electrode size is 12x12 micron
vert_el_dist = 0.02 
# There are 2 electrodes every 0.02 mm

# Mac
atlas = nib.load(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v2_pack/WHS_SD_rat_T2star_v1.01.nii.gz')
# Windows
# =============================================================================
# atlas_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas\WHS_SD_rat_atlas_v2_pack', 'WHS_SD_rat_T2star_v1.01.nii.gz')
# atlas = nib.load(atlas_path)
# =============================================================================
atlas_header = atlas.header
# get pixel dimension
pixdim = atlas_header.get('pixdim')[1]

# Labels
# Mac
labels_item = open(r"/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.label", "r")
# Windows
# labels_item = open(r"C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas\WHS_SD_rat_atlas_v4_beta.label", "r")
labels_index, labels_name, labels_color, labels_initial = readlabel( labels_item )   

# Segmentation
# Mac
segmentation = nib.load('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.nii.gz')
# Windows
#segmentation = nib.load(segmentation_path)
segmentation_data = segmentation.get_fdata()

# Probe colors
probe_colors = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']



path_probe_insertion = '/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Probe_Insertion'

# get the all the files in the probe folder
files_probe = os.listdir(path_probe_insertion)

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
P.append(pickle.load(open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Probe_Insertion/0probes.pkl', "rb")))
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
                    probe_y.append(PC[i][2])
                    probe_z.append(PC[i][1])
            elif P[k].Plane == 'c':
                for i in range(len(PC)):
                    probe_x.append(PC[i][2])
                    probe_y.append(PC[i][0])
                    probe_z.append(PC[i][1])  
            elif P[k].Plane == 'c':
                for i in range(len(PC)):        
                    probe_x.append(PC[i][0])
                    probe_y.append(PC[i][1])        
                    probe_z.append(PC[i][2])
            pts = np.array((probe_x, probe_y, probe_z)).T

            # fit the probe
            line_fit = Line.best_fit(pts)            
            # if no inclination in z direction
            if line_fit.direction[2] == 0:
                # line equations, to derive the starting and end point of the line (aka probe)
                # it is impartant that the first point clicked is the most external one, and the last correspond to the end of the probe
                z1 = pts[0,2]
                x1 = pts[0,0]
                y1 = line_fit.point[1]+((x1-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                z2 = pts[0,2]
                x2 = x2 = pts[-1,0]
                y2 = line_fit.point[1]+((x2-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]               
            else:
                # line equations, to derive the starting and end point of the line (aka probe)
                # it is impartant that the first point clicked is the most external one, and the last correspond to the end of the probe
                z1 = pts[0,2]
                x1 = line_fit.point[0]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                y1 = line_fit.point[1]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                z2 = pts[-1,2]
                x2 = line_fit.point[0]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                y2 = line_fit.point[1]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                # get the line to plot                
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
# =============================================================================
# Edges = np.load('Edges.npy')
# =============================================================================
mask = nib.load(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_brainmask_v1.01.nii.gz')
mask_data = mask.get_fdata()[:,:,:,0].transpose((2,1,0))
Edges = np.empty((512,1024,512))
for sl in range(0,1024):
    Edges[:,sl,:] = cv2.Canny(np.uint8((mask_data[:,sl,:])*255),100,200)  
edges = Edges.T

edges[:,::2,::] = edges[:,::2,::]*0
edges[:,::5,:] = edges[:,::5,:]*0
edges[::2,:,::] = edges[::2,:,::]*0
edges[:,:,::2] = edges[:,:,::2]*0

coords = np.array(np.where(edges == 255))
# Manage Points cloud
points = vedo.pointcloud.Points(coords)
# Create the mesh
mesh = Mesh(points)

# create some dummy data array to be associated to points
data = mesh.points()[:,2]  # pick z-coords, use them as scalar data
# build a custom LookUp Table of colors:
lut = buildLUT([
                (512, 'white', 0.07 ),
               ],
               vmin=0, belowColor='lightblue',
               vmax= 512, aboveColor='grey',
               nanColor='red',
               interpolate=False,
              )
mesh.cmap(lut, data)

dist = []

# To plot the probe with colors

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
# compute and display the insertion angle for each probe
for i in range(0,n):
    line_fit = getattr(LINE_FIT, color_used[i])
    deg_lat = math.degrees(math.atan(line_fit.direction[0]))
    deg_ant = math.degrees(math.atan(line_fit.direction[1]))
    print('\n\nAnalyze %s probe: \n ' %color_used[i])
    print('Estimated %s probe insertion angle: ' %color_used[i])
    print('%.2f degrees in the anterior direction' %deg_ant)
    print('%.2f degrees in the lateral direction\n' %deg_lat)

    # Get the brain regions traversed by the probe
    X1 = getattr(xyz, color_used[i])[0]
    X2 = getattr(xyz, color_used[i])[1]
    s = int(math.modf(X1[2])[1]) # starting point
    f = int(math.modf(X2[2])[1]) # ending point
    # get lenght of the probe
    dist.append(np.linalg.norm(f-s))
    regions = []
    colori = []
    initials = []
    index = []
    point_along_line = []
    if line_fit.direction[2] == 0:                
        for x in range(min(s,f), max(s,f)):                        
            y = line_fit.point[1]+((x-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
            z = pts[0,2]
            point_along_line.append([x,y,z])            
            if int(math.modf(x)[1])>512 or int(math.modf(y)[1])>1024 or int(math.modf(z)[1])>512:
                regions.append('Clear Label')
                colori.append('black')
                initials.append('CL')
            else:            
                regions.append(labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                colori.append(labels_color[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                initials.append(labels_initial[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
    else:
        for z in range(min(s,f),max(s,f)):
            x = line_fit.point[0]+((z-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
            y = line_fit.point[1]+((z-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
            point_along_line.append([x,y,z])
            if int(math.modf(x)[1])>512 or int(math.modf(y)[1])>1024 or int(math.modf(z)[1])>512:
                regions.append('Clear Label')
                colori.append('black')
                initials.append('CL')
            else:            
                regions.append(labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                colori.append(labels_color[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                initials.append(labels_initial[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
            #index.append(segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])])
    # count the number of elements in each region to 
    counter_regions = dict(Counter(regions))    
    regioni = list(OrderedDict.fromkeys(regions))
    iniziali = list(OrderedDict.fromkeys(initials))
    if 'Clear Label' in regioni:
        regioni.pop(regioni.index('Clear Label'))
        iniziali.pop(regioni.index('Clear Label'))
    cc = 0
    jj = 0
    num_el = []
    for re in regioni:
        # in the case in dont exit and then enter again the region
        position = [i for i,x in enumerate(regions) if x==re]
        # if there is only one point in the region
        if len(position) == 1:
            regional_dist = pixdim
        else:    
            # first point along the line in the region 
            start = [element * pixdim for element in point_along_line[position[0]]]
            # last point along the line in the region 
            end = [element * pixdim for element in point_along_line[position[-1]]]
            # length of the part of the probe in the region
            regional_dist = distance.euclidean(start,end)  
        # Number of electrodes in the region                
        num_el.append(round(regional_dist/vert_el_dist)*2)                                 
        #print(re)
        # proportion of the probe in the given region
        dist_prop = dist[i]/len(regioni)
        color_prop = labels_color[np.argwhere(np.array(labels_name)== re)]
        # plot the probe with the colors of the region traversed       
        ax1.add_patch(patches.Rectangle((100*i+5, cc), 17, dist_prop, color=color_prop[0][0]/255))
        plt.text(100*i, max(dist)+10, 'Probe %d\n(%s)'%(i+1, color_used[i]), color = color_used[i], fontsize=9, fontweight='bold')
        if len(iniziali[jj])>7:
            plt.text(100*i-2, cc+2, '%s-\n%s'%(iniziali[jj][0:5], iniziali[jj][6:]), fontsize=5.6)
        else:    
            plt.text(100*i-2, cc+4, '%s'%(iniziali[jj]), fontsize=6)
        plt.text(100*i+28, cc+4, '%d'%(num_el[jj]), fontsize=6.5)
        jj +=1
        cc = dist_prop*dist[i] + cc    
    #indici = list(OrderedDict.fromkeys(index))
    LL = [regioni,  iniziali, num_el]
    headers = [' Regions traversed', 'Initials', 'Channels']
    numpy_array = np.array(LL)
    transpose = numpy_array.T
    transpose_list = transpose.tolist()
    print(tabulate(transpose_list, headers, floatfmt=".2f"))
lims = (0,max(dist))
plt.ylim(lims)
plt.xlim((0,100*n+20))
plt.axis('off')


"""Build 2 windows that can interact and share functions"""
plt1 = Plotter(title='Brain viewer', size=(700,700), pos=(250,0))
# plot all the probes together
if n==1:
     plt1.show(mesh, getattr(pr,color_used[0]), getattr(L, color_used[0]), __doc__,
     axes=0, viewup="z", bg='black',
     )        
elif n==2:     
     plt1.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(L, color_used[0]), getattr(L, color_used[1]), __doc__,
     axes=0, viewup="z", bg='black',
     )
elif n == 3:
     plt1.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(L, color_used[0]),getattr(L, color_used[1]), getattr(L, color_used[2]), __doc__,
     axes=0, viewup="z",
     bg='black', 
     )
elif n == 4:
     plt1.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), __doc__,
     axes=0, viewup="z", bg='black',
     )
elif n==5:
     plt1.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), getattr(L, color_used[4]),  __doc__,
     axes=0, viewup="z", bg='black',
     )
elif n == 6:
     plt1.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(pr,color_used[5]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), getattr(L, color_used[4]), getattr(L, color_used[5]), __doc__,
     axes=0, viewup="z", bg='black',
     )






