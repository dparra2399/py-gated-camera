import os
import glob
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from felipe_utils.tof_utils_felipe import zero_norm_t
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, gaussian_filter1d

from spad_lib.spad512utils import (
    calculate_tof_domain_params,
    get_voltage_function,
    get_coarse_coding_matrix,
    get_hamiltonain_correlations,
    cluster_kmeans,
    build_coding_matrix_from_correlations,
    decode_depth_map,
    intrinsics_from_pixel_pitch,
    range_to_z
)

from spad_lib.file_utils import (
    get_data_folder,
    filter_npz_files,
    load_hot_mask,
    load_correlations_file,
    get_scheme_name
)

# -----------------------------------------------------------------------------
# CONFIG (capitalized)
# -----------------------------------------------------------------------------
PIXEL_PITCH = 16.38 #in uM
FOCAL_LENGTH = 25 #in mm
EXP = 6
K = 4
DATASET_TYPE = 'coarse'  # was `type`

N_TBINS_DEFAULT = 640
VMIN = 6.
VMAX = 7.5
MEDIAN_FILTER_SIZE = 5
CORRECT_MASTER = False
MASK_BACKGROUND_PIXELS = False
USE_CORRELATIONS = False
USE_FULL_CORRELATIONS = False
SIGMA_SIZE = 30
SHIFT_SIZE = 150
CORRECT_DEPTH_DISTORTION = False

# paths
TEST_FILE = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{EXP}/{DATASET_TYPE}k{K}_exp{EXP}.npz'
GT_FILE   = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{EXP}/{DATASET_TYPE}k{K}_gt_exp{EXP}.npz'
HOT_MASK_PATH = '/Users/davidparra/PycharmProjects/py-gated-camera/masks/hot_pixels.PNG'

EPSILON = 1e-12

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    hot_mask = load_hot_mask(HOT_MASK_PATH)

    gt_depth_map = None
    depth_map = None

    for path in [TEST_FILE, GT_FILE]:

        try:
            f = np.load(path)
        except FileNotFoundError:
            continue

        coded_vals = f["coded_vals"]
        total_time = f["total_time"]
        im_width = f["im_width"]
        freq = float(f["freq"])
        voltage = f["voltage"]
        try:
            size = f["size"]
        except KeyError:
            size = f['duty']
        n_tbins = int(f["n_tbins"])
        split_measurements = f["split_measurements"]

        # TOF params -------------------------------------------------------
        (   rep_tau,
             rep_freq,
             tbin_res,
             t_domain,
             max_depth,
             tbin_depth_res,
         ) = calculate_tof_domain_params(n_tbins, 1.0 / freq)
        mhz = int(freq * 1e-6)

        # -----------------------------------------------------------------
        # choose coding matrix depending on file type
        # -----------------------------------------------------------------
        if USE_CORRELATIONS:
            if 'coarse' in path:
                name_tmp = 'coarse'
            elif 'ham' in path:
                name_tmp = 'ham'
            else:
                assert False
            corr_path = (
                f"/Users/davidparra/PycharmProjects/py-gated-camera/correlation_functions/"
                f"{name_tmp}k{coded_vals.shape[-1]}_{mhz}mhz_{voltage}v_{size}w_correlations.npz"
            )
            correlations_total, n_tbins_corr = load_correlations_file(corr_path)
            (
                rep_tau,
                rep_freq,
                tbin_res,
                t_domain,
                max_depth,
                tbin_depth_res,
            ) = calculate_tof_domain_params(n_tbins_corr, 1.0 / freq)
            coding_matrix = build_coding_matrix_from_correlations(correlations_total, im_width, n_tbins_corr, freq,
                                                                  USE_FULL_CORRELATIONS, SIGMA_SIZE, SHIFT_SIZE)
            n_tbins = n_tbins_corr
        elif "coarse" in path:
            gate_width = f["gate_width"]

            irf = get_voltage_function(mhz, voltage, size, "pulse", n_tbins)
            coding_matrix = get_coarse_coding_matrix(
                gate_width * 1e3,
                coded_vals.shape[-1],
                0,
                gate_width * 1e3,
                rep_tau * 1e12,
                n_tbins,
                irf,
            )

        elif "ham" in path:
            if "pulse" in path:
                illum_type = "pulse"
            else:
                illum_type = "square"

            coding_matrix = get_hamiltonain_correlations(
                coded_vals.shape[-1], mhz, voltage, size, illum_type, n_tbins=n_tbins
            )
        else:
            assert False, 'Path needs to be "hamiltonian" or "coarse"'

        name = get_scheme_name(path, coded_vals.shape[-1])
        print(
            f"{name}\n\tVoltage: {voltage}\n\tSize: {size}\n\tTotal Time: {total_time}\n\tSplit Measurements: {split_measurements}"
        )

        # -----------------------------------------------------------------
        # decode to depth map
        # -----------------------------------------------------------------
        depth_map, zncc = decode_depth_map(
            coded_vals,
            coding_matrix,
            im_width,
            n_tbins,
            tbin_depth_res,
            USE_CORRELATIONS,
            USE_FULL_CORRELATIONS,
        )

        if CORRECT_DEPTH_DISTORTION:
            fx, fy, cx, cy = intrinsics_from_pixel_pitch(im_width, im_width, FOCAL_LENGTH, PIXEL_PITCH)
            depth_map = range_to_z(depth_map, fx, fy, cx, cy)

        # hot pixel fix
        filtered = median_filter(depth_map, size=3, mode="nearest")
        depth_map[hot_mask == 1] = filtered[hot_mask == 1]


        # optional crop for master
        if CORRECT_MASTER is False:
            depth_map = depth_map[:, : im_width // 2]

        # assign to GT or test map
        if path == GT_FILE:
            gt_depth_map = np.copy(depth_map)
            if MASK_BACKGROUND_PIXELS:
                mask = cluster_kmeans(np.copy(gt_depth_map), n_clusters=2)
                mask[mask == np.nanmax(mask)] = np.nan
                mask[mask == np.nanmin(mask)] = 1
        else:
            depth_map = np.copy(depth_map)
            coded_vals_save = np.copy(coded_vals)

    # apply mask to depth map if requested
    if MASK_BACKGROUND_PIXELS and gt_depth_map is not None:
        depth_map *= mask

    # ------------------------------------------------------------------
    # Diagnostics plots (coding vs coded at a few points)
    # ------------------------------------------------------------------
    # fallbacks if fewer than K points are given
    x1, y1 = (70, 70)
    x2, y2 = (220, 330)
    x3, y3 = (140, 240)
    points = [(x1, y1), (x2, y2), (x3, y3)]
    while len(points) < K:
        points.append((30, 330))

    norm_coded_vals_save = zero_norm_t(coded_vals_save, axis=-1)
    norm_coding_matrix_save = zero_norm_t(coding_matrix, axis=-1)

    fig, axs = plt.subplots(2, len(points)+1, figsize=(10, 5))
    axs[1, -1].imshow(median_filter(depth_map, size=MEDIAN_FILTER_SIZE), vmin=VMIN, vmax=VMAX)
    #axs[0, -1].set_axis_off()
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, (x, y) in enumerate(points):
        if USE_FULL_CORRELATIONS:
            axs[1, i].plot(norm_coding_matrix_save[y, x, :, :])
        else:
            axs[1, i].plot(norm_coding_matrix_save)
        for coded_val in norm_coded_vals_save[y, x, :]:
            axs[1, i].plot(depth_map[y, x] / tbin_depth_res, coded_val, 'o', color='red')
        axs[1, i].set_title(f'Correlations at {colors[i % len(colors)]} Dot')
        axs[1, -1].plot(x, y, 'o', color=colors[i % len(colors)])

        axs[0, i].bar(np.arange(0, K), coded_vals_save[y, x, :], color=colors[i % len(colors)])
        axs[0, i].set_title(f'Intensity Values \n Count = {np.sum(coded_vals_save[y, x, :])}')

    plt.show()

    # ------------------------------------------------------------------
    # Depth map + close-up + GT
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, K, height_ratios=[1, 1])

    for i in range(K):
        ax1 = fig.add_subplot(gs[0, i])
        filt = median_filter(np.copy(coded_vals_save[:, :, i]), size=3, mode='nearest')
        filt[hot_mask == 1] = filt[hot_mask == 1]
        filt = filt[:, :im_width // 2] if CORRECT_MASTER is False else filt
        im1 = ax1.imshow(filt, vmin=0, vmax=np.max(coded_vals_save) // 3)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Photon Counts')

    x, y = 20, 170
    width, height = 220, 320
    patch = depth_map[y:y+height, x:x+width]

    vmin_local = np.nanmin(depth_map) if VMIN is None else VMIN
    vmax_local = np.nanmax(depth_map) if VMAX is None else VMAX

    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(median_filter(depth_map, size=MEDIAN_FILTER_SIZE), vmin=vmin_local, vmax=vmax_local)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Depth (meters)')
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='lime', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title('Full Depth Map')

    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(median_filter(patch, size=MEDIAN_FILTER_SIZE), vmin=vmin_local, vmax=vmax_local)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Depth (meters)')
    ax3.set_title('Depth Map Closeup')

    if gt_depth_map is not None:
        ax4 = fig.add_subplot(gs[1, 2])
        im4 = ax4.imshow(gt_depth_map, vmin=vmin_local, vmax=vmax_local)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Depth (meters)')
        ax4.set_title('Ground Truth')

    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.05, wspace=0.05, hspace=0.05)
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    plt.show()

    print(f'min depth map: {np.min(depth_map)}, max depth map: {np.max(depth_map)}')


if __name__ == '__main__':
    main()
