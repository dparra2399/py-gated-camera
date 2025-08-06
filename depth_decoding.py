import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
from scipy import signal
import sys
from scipy.ndimage import gaussian_filter, median_filter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


import socket
import time
from PIL import Image
import numpy as np
import gated_project_code.spad_lib.metadata_constants as metadata_constants
from spad_lib.spad512utils import *


if __name__ == "__main__":
    cap_name = "acq00003"
    frame = 0
    img_num = 1
    name = f"IMG{img_num:05d}"
    base = f"/home/ubilaptop8/2025-Q2-David-P-captures/data/gated_images/{cap_name}"
    img_data = Image.open(f"{base}/{name}.PNG")
    img_data.load()

    metadata = metadata_constants.get_dict_metadata(img_data)

    gate_offset = metadata['Gate offset']
    gate_step_size = metadata['Gate step size']
    gate_steps = metadata['Gate steps']
    gate_width = metadata['Gate width']
    laser_time = metadata['Laser time']
    rep_tau = metadata['Laser frequency']

    n_tbins = 2000

    print(metadata)


    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1./ rep_tau)

    print(f'tbin_res = {tbin_res}')
    print(f'rep_tau = {rep_tau}')
    print(f'laser_time = {laser_time}')

    transient_img = np.zeros((512, 512, gate_steps))

    for i in range(gate_steps):
        name = f"IMG{i:05d}"
        img = Image.open(f"{base}/{name}.PNG")
        img.load()
        metadata_img = metadata_constants.get_dict_metadata(img)
        transient_img[:, :, i] = np.array(img)

    coding_matrix = get_coarse_coding_matrix(gate_step_size, gate_steps, gate_offset, gate_width, laser_time, n_tbins)

    norm_coding_matrix = zero_norm_t(coding_matrix)
    
    plt.imshow(coding_matrix.transpose(), aspect='auto')
    print(coding_matrix)
    plt.show()
    #exit(0)

    coded_vals = np.reshape(transient_img, (512 * 512, gate_steps))

    norm_coded_vals = zero_norm_t(coded_vals)

    print(norm_coded_vals.shape)
    print(norm_coding_matrix.shape)

    zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)
    
    depths = np.argmax(zncc, axis=-1)

    depth_map = np.reshape(depths, (512, 512)) * tbin_depth_res

    fig, axs = plt.subplots(3, figsize=(10, 10))

    x1, y1 = (320, 320)
    x2, y2 = (220, 320)

    axs[0].bar(np.arange(0, metadata['Gate steps']), transient_img[y1, x1, :], color='red')
    axs[1].bar(np.arange(0, metadata['Gate steps']), transient_img[y2, x2, :], color='blue')
    #axs[0].set_xticks(np.arange(0, metadata['Gate steps'])[::3])
    #axs[0].set_xticklabels(np.round(gate_starts, 1)[::3])


    axs[2].imshow(median_filter(depth_map, size=3))
    axs[2].plot(x1, y1, 'ro')
    axs[2].plot(x2, y2, 'bo')
    axs[2].set_title(cap_name)

    plt.show()
    print('done')