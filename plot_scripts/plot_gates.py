import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
from scipy import signal
import sys
import socket
import time
from PIL import Image
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import gated_project_code.spad_lib.metadata_constants as metadata_constants


'''
This visualized the gate sequences for a single frame accross the laser period.
'''

if __name__ == "__main__":
    cap_name = "acq00004"
    frame = 0
    img_num = 1
    name = f"IMG{img_num:05d}"
    base = f"/home/ubilaptop8/2025-Q2-David-P-captures/data/gated_images/{cap_name}"
    img_data = Image.open(f"{base}/{name}.PNG")
    img_data.load()

    metadata = metadata_constants.get_dict_metadata(img_data) #Read metadata
    print(metadata)

    gate_offset = metadata['Gate offset']
    gate_step_size = metadata['Gate step size']
    gate_steps = metadata['Gate steps']

    #Calculate each gate starting time
    gate_starts = np.array([gate_offset + (gate_step_size * (gate_step)) for gate_step in range(gate_steps)])

    #fig, axs = plt.subplots(gate_steps, figsize=(10, 10))

    '''
    Plot each gate foor number of gate steps
    '''
    time_resolution = np.linspace(0, metadata['Laser time'], 1000)
    coding_matrix = np.zeros((1000, gate_steps))
    for gate_step in range(gate_steps):
        gate_start = gate_starts[gate_step]
        gate_end = gate_start + metadata['Gate width']
        
        #Make gate start and end 
        start_idx = np.searchsorted(time_resolution, gate_start, side='left')
        end_idx = np.searchsorted(time_resolution, gate_end, side='right')

        indices = np.arange(start_idx, end_idx)

        gate = np.zeros_like(time_resolution)
        gate[indices] = 1

        coding_matrix[:, gate_step] = gate

        #axs[gate_step].plot(gate)
        #axs[gate_step].set_xticks(np.arange(0, 1000)[::100])
        #axs[gate_step].set_xticklabels([])
        #axs[gate_step].set_yticklabels([])
    #axs[-1].set_xticklabels(np.round(time_resolution, 1)[::100])

    plt.imshow(np.repeat(coding_matrix.transpose(), 100, axis=0), aspect='auto', cmap='gray')
    plt.show()
