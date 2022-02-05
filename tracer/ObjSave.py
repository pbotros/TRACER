#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:36:26 2020

@author: jacopop
"""
import base64
import os
from dataclasses import dataclass, field
from io import StringIO, BytesIO
from typing import Tuple, List, Optional

import dataclasses_json
import marshmallow
import numpy as np
from dataclasses_json import dataclass_json, DataClassJsonMixin


class save_transform(object):
    def __init__(self, tracker_slice, tracker_slice_angle, coord, image, nolines):
        self.Slice = tracker_slice
        self.SliceAngle = tracker_slice_angle
        self.Transform_points = coord
        self.Transform = image
        self.Transform_withoulines = nolines  
        
# # Store probe features
# YIKES THIS IS TERRIBLE??
class save_probe(object):
    def __init__(self, slice, slice_angle, probe_name, plane, probe_counter):
        self.Slice = slice
        self.SliceAngle = slice_angle
        self.Probe = probe_name
        self.Plane = plane
        self.Counter = probe_counter
        
class save_probe_insertion(object):
    def __init__(self, coord, plane, probe_counter):
        self.Probe = coord
        self.Plane = plane
        self.Counter = probe_counter        
        
        
# object for the clicked probes
class probe_obj(object):
    pass

def _encode_np(arr):
    io = BytesIO()
    np.save(io, arr)
    io.seek(0)
    return base64.b64encode(io.read()).decode('ASCII')

def _decode_np(b):
    io = BytesIO()
    io.write(base64.b64decode(b))
    io.seek(0)
    return np.load(io, allow_pickle=True)


@dataclass
class HistologyTransform(DataClassJsonMixin):
    histology_filename: str
    transform_atlas_plane: str
    transform_points_histology: List[Tuple[int, int]] = field(default_factory=list)
    transform_points_atlas: List[Tuple[int, int]] = field(default_factory=list)
    transform_atlas_slice_index: Optional[int] = None
    transform_atlas_angle: float = 0.0

    img_warped: Optional[np.ndarray] = field(
        default=None,
        metadata=dataclasses_json.config(
            encoder=_encode_np,
            decoder=_decode_np,
            mm_field=marshmallow.fields.String()
        ))
    img_warped_overlay: Optional[np.ndarray] = field(
        default=None,
        metadata=dataclasses_json.config(
            encoder=_encode_np,
            decoder=_decode_np,
            mm_field=marshmallow.fields.String()
        ))

    @property
    def histology_basename_noext(self) -> str:
        return os.path.splitext(os.path.basename(self.histology_filename))[0]


@dataclass
class ProbeTrace(DataClassJsonMixin):
    histology_transform: HistologyTransform

    probe_name: str
    color: str

    # List of 3D coordinates in atlas space
    probe_atlas_coords: List[Tuple[int, int, int]] = field(default_factory=list)

    @property
    def histology_basename_noext(self) -> str:
        return self.histology_transform.histology_basename_noext

