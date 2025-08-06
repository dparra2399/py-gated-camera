#!/usr/bin/env python

import sys
import socket
import time
from PIL import Image

# Path to the SPAD512S image
cap_name = "acq00004"
frame = 2
img_num = 1
name = f"IMG{img_num:05d}"
base = f"/home/ubilaptop8/2025-Q2-David-P-captures/data/gated_images/{cap_name}"
im = Image.open(f"{base}/{name}-{frame:04d}.PNG")
im.load()

# Metadata keys to check and print
metadata_keys = [
    'Author', 'System', 'Date taken', 'Time taken', 'Mode',
    'Integration time', 'Laser frequency', 'Overlap', 'Frame', 
    'Frames', 'Gate step', 'Gate steps', 'Gate step arbitrary', 
    'Gate step size', 'Gate width', 'Gate offset', 'Gate increment', 
    'External frame trigger', 'External gate trigger', 'Software version'
]

# Print metadata if available
for key in metadata_keys:
    value = im.info.get(key)  # Use .get() to safely access the key
    if value is not None:  # Only print if the key exists
        print(f'{key}: {value}')