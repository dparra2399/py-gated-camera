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
import math

correct_master = False
exp = 12
k = 4
n_tbins = 1_024
vmin = 19
vmax = 21
mask_background_pixels = True
try:
    #filename = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{exp}/coarsek{k}_exp{exp}.npz'
    test_file = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{exp}/hamK{k}_exp{exp}.npz'
    #filename = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{exp}/coarse_gt_exp{exp}.npz'

    gt_file = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{exp}/coarse_gt_exp{exp}.npz'
except FileNotFoundError:
    assert False, 'file not found'


filenames = [test_file, gt_file]

for filename in filenames:

    file = np.load(filename)

    coded_vals = file['coded_vals']
    if 'gt' not in filename:
        coded_vals_save = np.copy(coded_vals)
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

    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1 / freq)
    mhz = int(freq * 1e-6)

    if 'coarse' in filename:
        gate_width = file["gate_width"]
        if num_gates == 3:
            size = 34
        elif num_gates == 4:
            size = 25
        else:
            size = 12
        irf = get_voltage_function(mhz, voltage, size, 'pulse', n_tbins)
        coding_matrix = get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0, gate_width * 1e3, rep_tau * 1e12, n_tbins, irf)
        # plt.imshow(coding_matrix.transpose(), aspect='auto')
        # plt.show()
    elif 'ham' in filename:
        K = coded_vals.shape[-1]
        if 'pulse' in filename:
            illum_type = 'pulse'
            size = 12
            voltage = 10

        else:
            illum_type = 'square'
            size = 20

        coding_matrix = get_hamiltonain_correlations(K, mhz, voltage, size, illum_type, n_tbins=n_tbins)
    else:
        exit(0)

    norm_coding_matrix = zero_norm_t(coding_matrix)

    norm_coded_vals = zero_norm_t(coded_vals)

    print(norm_coded_vals.shape)
    print(norm_coding_matrix.shape)

    zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)

    if correct_master:
        zncc[:, im_width // 2:, :] = np.roll(zncc[:, im_width //2:, :], shift=870)


    depths = np.argmax(zncc, axis=-1)

    if 'gt' in filename:
        gt_depth_map = np.reshape(depths, (512, 512)) * tbin_depth_res
    else:
        depth_map = np.reshape(depths, (512, 512)) * tbin_depth_res


    if 'gt' in filename and mask_background_pixels:
        tmp = cluster_kmeans(np.copy(gt_depth_map), n_clusters=2)
        tmp[tmp == np.nanmax(tmp)] = np.nan
        tmp[tmp == np.nanmin(tmp)] = 1

if mask_background_pixels:
    depth_map *= tmp

x1, y1 = (70, 70)
x2, y2 = (220, 330)

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])  # 2 rows, 2 cols

ax0 = fig.add_subplot(gs[0, 0])
ax0.bar(np.arange(0, coded_vals_save.shape[-1]), coded_vals_save[y1, x1, :], color='red')
ax0.set_title(f'Sample Intensity Values (Red Dot), Count = {np.sum(coded_vals_save[y1, x1, :])}')
ax1 = fig.add_subplot(gs[0, 1])
ax1.bar(np.arange(0, coded_vals_save.shape[-1]), coded_vals_save[y2, x2, :], color='blue')
ax1.set_title(f'Sample Intensity Values (Blue Dot),  Count = {np.sum(coded_vals_save[y2, x2, :])}')

#Here are parameters to show a box close up of the depth map for better visualization
#Need to tune yourself
x, y = 20, 170
width, height = 220, 320


patch = depth_map[y:y+height, x:x+width]

ax2 = fig.add_subplot(gs[1, 0])
if vmin is None:
    vmin = np.nanmin(depth_map)
if vmax is None:
    vmax = np.nanmax(depth_map)
if correct_master:
    im2 = ax2.imshow(depth_map, vmin=vmin, vmax=vmax)
else:
    #ax2.imshow(gaussian_filter(median_filter(depth_map[:, :im_width // 2], size=7), sigma=10))
    im2 = ax2.imshow(depth_map[:,:im_width//2], vmin=vmin, vmax=vmax)

fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Depth (meters)')  # Add colorbar for ax2

ax2.plot(x1, y1, 'ro')
ax2.plot(x2, y2, 'bo')

rect = patches.Rectangle((x, y), width, height,
                         linewidth=2, edgecolor='lime', facecolor='none')
ax2.add_patch(rect)
ax2.set_title('Full Depth Map')


ax3 = fig.add_subplot(gs[1, 1])
im3 = ax3.imshow(patch, vmin=vmin, vmax=vmax)
fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04,  label='Depth (meters)')  # Add colorbar for ax2

#error = np.mean(np.abs(patch - np.mean(patch)))
#ax3.set_xlabel(f'MAE: {error*1000: .3f} mm')
ax3.set_title('Depth Map Closeup')

plt.tight_layout()
plt.show()
print(f'min depth map: {np.min(depth_map)}, max depth map: {np.max(depth_map)}')


