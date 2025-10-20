import os
import sys
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
from PIL import Image
import math
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse

exp_num = 9
k = 3
n_tbins = 640
vmin = 21.09
vmax = 27
median_filter_size = 1
correct_master = False
mask_background_pixels = True
folder = f"/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{exp_num}"
#folder = f"/mnt/researchdrive/research_users/David/Gated_Camera_Project/gated_project_data/exp{exp_num}"

#hot_mask_filename = '/home/ubi-user/David_P_folder/py-gated-camera/masks/hot_pixels.PNG'
hot_mask_filename = '/Users/davidparra/PycharmProjects/py-gated-camera/masks/hot_pixels.PNG'

#hot_mask = np.load(hot_mask_filename)
hot_mask = np.array(Image.open(hot_mask_filename))
hot_mask[hot_mask < 5000] = 0
hot_mask[hot_mask > 0] = 1

npz_files = glob(os.path.join(folder, "*.npz"))

npz_files = [f for f in npz_files if (str(k) in os.path.basename(f) or "_gt_" in os.path.basename(f))
             and 'pulse' not in os.path.basename(f)]

depths_maps_dict = {}
#depths_maps_normalized = []
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
    try:
        split_measurements = file["split_measurements"]
    except:
        split_measurements = False

    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1 / freq)
    mhz = int(freq * 1e-6)
    if path == npz_files[0]:
        print(f'MHZ: {mhz} \n Max Depth: {max_depth}')
        print()

    if 'coarse' in path:
        gate_width = file["gate_width"]
        try:
            size = file["size"]
        except:
            if num_gates == 3 and mhz == 10:
                voltage = 7
                size = 34
            elif num_gates == 3 and mhz == 5:
                voltage = 5.7
                size = 67
            elif num_gates == 4 and mhz == 10:
                voltage = 7.6
                size = 25
            elif num_gates == 4 and mhz == 5:
                voltage = 6
                size = 50
            else:
                voltage = 10
                size = 12

        irf = get_voltage_function(mhz, voltage, size, 'pulse', n_tbins)
        #if num_gates == 3:
        #    irf = gaussian_pulse(np.arange(n_tbins), 0, 180, circ_shifted=True)
        #    irf = np.roll(irf, shift=100)
        #plt.plot(irf)
        #plt.show()
        # irf = np.roll(irf, -np.argmax(irf))
        # irf2 = gaussian_pulse(np.arange(n_tbins), 0, 100, circ_shifted=True)
        # irf2 = np.roll(irf2, np.argmax(irf2))
        #plt.plot(irf)
        # plt.plot(irf2)
        # plt.title('Irf Function')
        #plt.show()
        coding_matrix = get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0,
                                                 gate_width * 1e3, rep_tau * 1e12,
                                                 n_tbins=n_tbins, irf=irf)
        if 'gt' in path:
            name = 'GroundTruth'
        else:
            name = 'CoarseK{}'.format(num_gates)
            #plt.plot(coding_matrix)
            #plt.show()
    elif 'ham' in path:
        K = coded_vals.shape[-1]


        if 'pulse' in path:
            illum_type = 'pulse'
            size = 12
            voltage = 10
            name = 'HamiltonianK{} (Pulsed)'.format(K)

        else:
            illum_type = 'square'
            size = 20
            name = 'HamiltonianK{}'.format(K)

        coding_matrix = get_hamiltonain_correlations(K, mhz, voltage, size, illum_type, n_tbins=n_tbins)
        plt.plot(coding_matrix)
        plt.show()
    else:
        assert False, 'Path needs to be "hamiltonian" or "coarse".'

    print(f'{name} \n\t Voltage: {voltage} \n\t' +
          f' Size: {size} \n\t Total Time: {total_time} \n\t Split Measurements: {split_measurements}')
    print()
    #plt.plot(coding_matrix)
    #plt.show()
    norm_coding_matrix = zero_norm_t(coding_matrix)

    norm_coded_vals = zero_norm_t(coded_vals)

    zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)

    if correct_master:
        zncc[:, im_width // 2:, :] = np.roll(zncc[:, im_width // 2:, :], shift=175)

    #zncc = np.roll(zncc, shift=5)
    depths = np.argmax(zncc, axis=-1)

    depth_map = np.reshape(depths, (512, 512)) * tbin_depth_res

    filtered = median_filter(depth_map, size=3, mode='nearest')
    depth_map[hot_mask == 1] = filtered[hot_mask == 1]

    if name == 'GroundTruth' and mask_background_pixels:
        tmp = cluster_kmeans(np.copy(depth_map), n_clusters=2)
        tmp[tmp == np.nanmax(tmp)] = np.nan
        tmp[tmp == np.nanmin(tmp)] = 1
        depth_map *= tmp
        # plt.imshow(depth_map, cmap='gray')
        # plt.show()
        coded_vals_tmp = np.sum(coded_vals, axis=-1)
        #depth_map[coded_vals_tmp < 2500] = np.nan

        #vmin = max(np.nanmin(depth_map), vmin)
        vmin = 21.1
        #vmax = np.nanmax(depth_map) + 0.5
        vmax = 21.15


    depths_maps_dict[name] = depth_map
    #depth_map_normalized = (depth_map - np.nanmean(depth_map)) / np.nanstd(depth_map)

    #depths_maps.append(depth_map)
    #depths_maps_normalized.append(depth_map_normalized)

#depth_maps = np.stack(depths_maps, axis=-1)
#depths_maps_normalized = np.stack(depths_maps_normalized, axis=-1)
gt_depth_map = depths_maps_dict.pop('GroundTruth', None)

print(f'Min depth: {np.nanmin(gt_depth_map)}, Max depth: {np.nanmax(gt_depth_map)}')

x, y = 20, 170
width, height = 220, 320
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, len(depths_maps_dict)+1, height_ratios=[1, 2])  # 2 rows, 2 cols

for i in range(len(depths_maps_dict)+1):

    if i == len(depths_maps_dict) and gt_depth_map is not None:
        name = 'GroundTruth'
        depth_map = gt_depth_map
    elif gt_depth_map is not None and mask_background_pixels:
        name, depth_map = list(depths_maps_dict.items())[i]
        mask = np.copy(gt_depth_map)
        mask[~np.isnan(mask)] = 1
        #mask[np.isnan(gt_depth_map)] = 0
        depth_map = depth_map * mask
    elif gt_depth_map is not None:
        name, depth_map = list(depths_maps_dict.items())[i]
    else:
        break

    #depth_map_normalized = depths_maps_normalized[:, :, i]

    #patch_normalized = depth_map_normalized[y:y+height, x:x+width]
    patch = depth_map[y:y+height, x:x+width]
    ax = fig.add_subplot(gs[0, i])
    if correct_master:
        im = ax.imshow(median_filter(depth_map, size=median_filter_size), vmin=vmin, vmax=vmax)
    else:
        #ax.imshow(gaussian_filter(median_filter(depth_map[:, :im_width // 2], size=5), sigma=0.0),
        #          vmin=np.nanmin(depth_maps), vmax=np.nanmax(depth_maps))
        #ax.imshow(depth_map[:, :im_width//2], vmin=np.nanmin(depth_maps), vmax=np.nanmax(depth_maps))
        if gt_depth_map is not None:
            #ax.imshow(depth_map[:, :im_width//2], vmin=np.nanmin(gt_depth_map), vmax=np.nanmax(gt_depth_map))
            im = ax.imshow(gaussian_filter(median_filter(depth_map[:, :im_width // 2], size=median_filter_size), sigma=0.0),  vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(median_filter(depth_map[:, :im_width // 2], size=median_filter_size),  vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Depth (meters)')  # Add colorbar for ax
    ax.set_title(name)
    rect = patches.Rectangle((x, y), width, height,
                             linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

    ax2 = fig.add_subplot(gs[1, i])
    if gt_depth_map is not None:
        im2 = ax2.imshow(median_filter(patch[:, :im_width // 2], size=median_filter_size), vmin=vmin, vmax=vmax)
    else:
        im2 = ax2.imshow(median_filter(patch[:, :im_width // 2], size=median_filter_size),  vmin=vmin, vmax=vmax)

    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Depth (meters)')  # Add colorbar for ax2

    if gt_depth_map is not None:
        error = np.nanmean(np.abs(depth_map[:, :im_width // 2] - gt_depth_map[:, :im_width // 2]))
        rmse = np.sqrt(np.nanmean((depth_map[:, :im_width // 2]  - gt_depth_map[:, :im_width // 2])**2))
        if i != len(depths_maps_dict):
            ax2.set_xlabel(f'MAE: {error*1000: .3f} mm\n RMSE: {rmse*1000: .3f} mm')

plt.tight_layout()
plt.show()