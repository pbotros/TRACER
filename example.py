#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:05:20 2021

@author: pearlsaldanha
"""

import os 
os.chdir("/Users/admin")
from TRACER import *

atlas = AtlasLoader(atlas_folder='/Users/admin/TRACER/waxholm_atlas', atlas_version ='v4')
presurgery = ProbesInsertion(atlas, probe_folder ='/Users/admin/TRACER/probes')

presurgery.re_load_probes('Probe0')
probe_folder = '/Users/admin/TRACER/probes'
vis3d_presurgery = vis_inserted_probes(atlas, probe_folder)
vis3d_presurgery.vis2d()
vis3d_presurgery.vis3d()

preprocess_hist = preprocess_histology('/Users/admin/TRACER/histology')

register_probes = ProbesRegistration(atlas, processed_histology_folder='/Users/admin/TRACER/histology/processed', show_hist=True)
probe_folder = '/Users/admin/TRACER/histology/processed/probes'
vis3dres = vis_registered_probes(atlas,probe_folder)
vis3dres.vis3d()
vis3dres.vis2d()