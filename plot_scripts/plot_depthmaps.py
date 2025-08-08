import os
import glob

from spad_lib.SPAD512S import SPAD512S
from spad_lib.spad512utils import *
import numpy as np
import time
import matplotlib.pyplot  as plt
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter, median_filter
from felipe_utils import CodingFunctionsFelipe
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from glob import glob
import math

exp_num = 1
n_tbins = 1000
plot_fig = True
correct_master = False
folder = f"/Volumes/velten/Research_Users/David/gated_project_data/exp{exp_num}"

npz_files = glob(os.path.join(folder, "*.npz"))

depths_maps = []
depths_maps_normalized = []
for path in npz_files:
    file = np.load(path)

    coded_vals = file['coded_vals']
    irf = file['irf']
    total_time = file["total_time"]
    num_gates = file["num_gates"]
    im_width = file["im_width"]
    bitDepth = file["bitDepth"]
    iterations = file["iterations"]
    overlap = file["overlap"]
    timeout = file["timeout"]
    pileup = file["pileup"]
    gate_steps = file["gate_steps"]
    gate_step_arbitrary = file["gate_step_arbitrary"]
    gate_step_size = file["gate_step_size"]
    gate_offset = file["gate_offset"]
    gate_direction = file["gate_direction"]
    gate_trig = file["gate_trig"]
    voltage = file["voltage"]
    freq = file["freq"]
    gate_width = file["gate_width"]

    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1 / freq)
    mhz = int(freq * 1e-6)

    if 'coarse' in path:
        irf = get_voltage_function(mhz, voltage, 'pulse', n_tbins=n_tbins)
        #irf = None
        coding_matrix = get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0,
                                                 gate_width * 1e3, rep_tau * 1e12,
                                                 n_tbins=n_tbins, irf=irf)
    elif 'ham' in path:
        K = coded_vals.shape[-1]
        coding_matrix = get_hamiltonain_correlations(K, mhz, voltage, n_tbins=n_tbins)

    else:
        assert False, 'Path needs to be "hamiltonian" or "coarse".'

    norm_coding_matrix = zero_norm_t(coding_matrix)

    norm_coded_vals = zero_norm_t(coded_vals)

    zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)

    if correct_master:
        zncc[:, im_width // 2:, :] = np.roll(zncc[:, im_width // 2:, :], shift=-5)

    #zncc = np.roll(zncc, shift=5)
    depths = np.argmax(zncc, axis=-1)

    depth_map = np.reshape(depths, (512, 512)) * tbin_depth_res
    depth_map_normalized = (depth_map - np.nanmean(depth_map)) / np.nanstd(depth_map)

    depths_maps.append(depth_map)
    depths_maps_normalized.append(depth_map_normalized)

depth_maps = np.stack(depths_maps, axis=-1)
depths_maps_normalized = np.stack(depths_maps_normalized, axis=-1)
if plot_fig:
    x, y = 20, 170
    width, height = 220, 320
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, depth_maps.shape[-1], height_ratios=[1, 2])  # 2 rows, 2 cols

    for i in range(depth_maps.shape[-1]):
        depth_map = depth_maps[:, :, i]
        #depth_map_normalized = depths_maps_normalized[:, :, i]

        #patch_normalized = depth_map_normalized[y:y+height, x:x+width]
        patch = depth_map[y:y+height, x:x+width]
        ax = fig.add_subplot(gs[0, i])
        if correct_master:
            ax.imshow(depth_map, vmin=np.nanmin(depth_map), vmax=np.nanmax(depth_map))
        else:
            ax.imshow(gaussian_filter(median_filter(depth_map[:, :im_width // 2], size=5), sigma=0.0),
                      vmin=np.nanmin(depth_maps), vmax=np.nanmax(depth_maps))
            #ax.imshow(depth_map[:, :im_width//2], vmin=np.nanmin(depth_maps), vmax=np.nanmax(depth_maps))

        if 'coarse' in npz_files[i]:
            ax.set_title('Coarse Histogram')
        elif 'ham' in npz_files[i]:
            ax.set_title(f'Hamiltonian K{np.load(npz_files[i])["coded_vals"].shape[-1]}')
        rect = patches.Rectangle((x, y), width, height,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(patch, vmin=np.nanmin(depth_maps), vmax=np.nanmax(depth_maps))

        error = np.mean(np.abs(patch - np.mean(patch)))
        ax2.set_xlabel(f'MAE: {error*1000: .3f} mm')

    plt.tight_layout()
    plt.show()